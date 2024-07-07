import sys

sys.path.append("./")

from typing import Tuple

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
)
from model_util import load_models_xl, get_torch_device, torch_gc
from controlnet_util import openpose, get_depth_map, get_canny_image

import wandb


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# We  are building a script that takes a json of parameters and tries all combinations of them
def draw_kps(
    image_pil,
    kps,
    color_list=[
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ],
):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image


class InstantIDModelPipe:
    def __init__(self, xformers=False):
        # initalize the components inherent to the Identity model
        self.xformers = xformers
        if xformers:
            self.app = FaceAnalysis(
                name="antelopev2",
                root="./",
                providers=["CPUExecutionProvider"],
            )
        else:
            self.app = FaceAnalysis(
                name="antelopev2",
                root="./",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        self.device = get_torch_device()
        self.dtype = (
            torch.float16 if str(self.device).__contains__("cuda") else torch.float32
        )

        # Path to InstantID models
        self.face_adapter_path = f"./checkpoints/ip-adapter.bin"
        self.controlnet_path = f"./checkpoints/ControlNetModel"

        # Load pipeline face ControlNetModel
        self.controlnet_identitynet = ControlNetModel.from_pretrained(
            self.controlnet_path, torch_dtype=self.dtypedtype
        )
        self.pipe, self.current_model, self.current_lora = None, None, None

        # TODO: Refactor to remove controlnets
        # controlnet-pose
        controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
        controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
        controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

        controlnet_pose = ControlNetModel.from_pretrained(
            controlnet_pose_model, torch_dtype=self.dtype
        ).to(self.device)
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_canny_model, torch_dtype=self.dtype
        ).to(self.device)
        controlnet_depth = ControlNetModel.from_pretrained(
            controlnet_depth_model, torch_dtype=self.dtype
        ).to(self.device)

        self.controlnet_map = {
            "pose": controlnet_pose,
            "canny": controlnet_canny,
            "depth": controlnet_depth,
        }
        self.controlnet_map_fn = {
            "pose": openpose,
            "canny": get_canny_image,
            "depth": get_depth_map,
        }

    def _initalize_pipeline(self, pretrained_model_name_or_path, lora_path=None):
        if pretrained_model_name_or_path.endswith(
            ".ckpt"
        ) or pretrained_model_name_or_path.endswith(".safetensors"):
            scheduler_kwargs = hf_hub_download(
                repo_id="wangqixun/YamerMIX_v8",
                subfolder="scheduler",
                filename="scheduler_config.json",
            )

            (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                scheduler_name=None,
                weight_dtype=self.dtype,
            )

            scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
            pipe = StableDiffusionXLInstantIDPipeline(
                vae=vae,
                text_encoder=text_encoders[0],
                text_encoder_2=text_encoders[1],
                tokenizer=tokenizers[0],
                tokenizer_2=tokenizers[1],
                unet=unet,
                scheduler=scheduler,
                controlnet=[self.controlnet_identitynet],
            ).to(self.device)

        else:
            pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                pretrained_model_name_or_path,
                controlnet=[self.controlnet_identitynet],
                torch_dtype=self.dtype,
                safety_checker=None,
                feature_extractor=None,
            ).to(self.device)

            pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )

        if self.xformers:
            pipe.cuda(use_xformers=True)
            pipe.enable_xformers_memory_efficient_attention()

        pipe.load_ip_adapter_instantid(self.face_adapter_path)
        # load and disable LCM
        # TODO: Refactor this to support custom LORAs
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        pipe.disable_lora()

        return pipe, pretrained_model_name_or_path, lora_path

    def _generate_single_image(
        self,
        face_image_path,
        pose_image_path,
        prompt,
        negative_prompt,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
        seed=42,
        scheduler="EulerDiscreteScheduler",
        enable_LCM=False,
        enhance_face_region=False,
        face_detect_threshold=0.50,
    ):
        assert self.pipe is not None, "Please initalize the pipeline first"

        if enable_LCM:
            self.pipe.scheduler = diffusers.LCMScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.pipe.enable_lora()
        else:
            self.pipe.disable_lora()
            scheduler_class_name = scheduler.split("-")[0]

            add_kwargs = {}
            if len(scheduler.split("-")) > 1:
                add_kwargs["use_karras_sigmas"] = True
            if len(scheduler.split("-")) > 2:
                add_kwargs["algorithm_type"] = "sde-dpmsolver++"
            scheduler = getattr(diffusers, scheduler_class_name)
            self.pipe.scheduler = scheduler.from_config(
                self.pipe.scheduler.config, **add_kwargs
            )

        self.app.prepare(
            ctx_id=0, det_thresh=face_detect_threshold, det_size=(640, 640)
        )

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image, max_side=1024)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = self.app.get(face_image_cv2)

        if len(face_info) == 0:
            raise ValueError(
                f"Unable to detect a face in the image. Please upload a different photo with a clear face."
            )
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[
            -1
        ]  # only use the maximum face
        # crop the face
        face_cropped = face_image.crop(face_info["bbox"])

        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
        img_controlnet = face_image
        if pose_image_path is not None:
            pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image, max_side=1024)
            img_controlnet = pose_image
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = self.app.get(pose_image_cv2)

            if len(face_info) == 0:
                raise ValueError(
                    f"Cannot find any face in the reference image! Please upload another person image"
                )

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None

        if len(controlnet_selection) > 0:
            controlnet_scales = {
                "pose": pose_strength,
                "canny": canny_strength,
                "depth": depth_strength,
            }
            self.pipe.controlnet = MultiControlNetModel(
                [self.controlnet_identitynet]
                + [self.controlnet_map[s] for s in controlnet_selection]
            )
            control_scales = [float(identitynet_strength_ratio)] + [
                controlnet_scales[s] for s in controlnet_selection
            ]
            control_images = [face_kps] + [
                self.controlnet_map_fn[s](img_controlnet).resize((width, height))
                for s in controlnet_selection
            ]
        else:
            self.pipe.controlnet = self.controlnet_identitynet
            control_scales = float(identitynet_strength_ratio)
            control_images = face_kps

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        self.pipe.set_ip_adapter_scale(adapter_strength_ratio)
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images
        return images[0], face_cropped

    def generate_subjects_image_grid(
        self,
        subject_image_path_list,
        pose_image_path,
        prompt,
        negative_prompt,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        pose_strength,
        canny_strength,
        depth_strength,
        controlnet_selection,
        guidance_scale,
    ):
        # make sure all the images exist

        for image_path in subject_image_path_list:
            if not os.path.exists(image_path):
                raise ValueError(f"Image {image_path} does not exist")

        # call the generate_single_image for each image
        # and then create a grid of images
        images = []
        face_images = []
        for image_path in subject_image_path_list:
            image, face_image = self._generate_single_image(
                image_path,
                pose_image_path,
                prompt,
                negative_prompt,
                num_steps,
                identitynet_strength_ratio,
                adapter_strength_ratio,
                pose_strength,
                canny_strength,
                depth_strength,
                controlnet_selection,
                guidance_scale,
            )
            images.append(image)
            face_images.append(face_image)

        # create a grid of images with the face images and the generated images next to them

        # resize the face images to the same size
        face_images_resized = []
        for face_image in face_images:
            face_images_resized.append(resize_img(face_image, size=(256, 256)))

        # resize the generated images to the same size
        images_resized = []
        for image in images:
            images_resized.append(resize_img(image, size=(256, 256)))

        # create a grid image with up to 5 images on each column and the face image on the left
        grid_image = None

        for i in range(0, len(images_resized), 5):
            images_row = images_resized[i : i + 5]
            face_images_row = face_images_resized[i : i + 5]

            row_images = [face_images_row[0]]
            row_images.extend(images_row)

            row_images_pil = [convert_from_cv2_to_image(img) for img in row_images]
            row_images_pil = [img.convert("RGB") for img in row_images_pil]

            widths, heights = zip(*(i.size for i in row_images_pil))

            total_width = sum(widths)
            max_height = max(heights)

            grid_image_row = Image.new("RGB", (total_width, max_height))

            x_offset = 0
            for img in row_images_pil:
                grid_image_row.paste(img, (x_offset, 0))
                x_offset += img.size[0]

            if grid_image is None:
                grid_image = grid_image_row
            else:
                grid_image = Image.new(
                    "RGB", (grid_image.width, grid_image.height + grid_image_row.height)
                )
                grid_image.paste(
                    grid_image_row, (0, grid_image.height - grid_image_row.height)
                )

        return grid_image
