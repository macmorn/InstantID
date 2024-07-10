import os
import sys

sys.path.append("./")
from gird_search_images import InstantIDModelPipe


def test_single_image_generation():
    # assert that te image exists
    assert os.path.exists("examples/musk_resize.jpeg")
    model = InstantIDModelPipe(xformers=False)
    model._initalize_pipeline(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0")
    image, cropped_face = model._generate_single_image(
        face_image_path="examples/musk_resize.jpeg",
        pose_image_path=None,
        prompt="A portrait of a person in a clown costume",
        negative_prompt="",
        num_steps=15,
        guidance_scale=5,
        identitynet_strength_ratio=0.5,
        adapter_strength_ratio=0.5,
        pose_strength=0,
        canny_strength=0,
        depth_strength=0,
        controlnet_selection=[],
    )
    assert image is not None
    # save the image
    image.save("examples/musk_clown.jpg")

def test_grid_creation():
    subject_paths =  [
        "examples/kaifu_resize.png",
        "examples/sam_resize.png",
        "examples/yann-lecun_resize.jpg",
        "examples/schmidhuber_resize.png",
        "examples/kaifu_resize.png",
        "examples/sam_resize.png",
        "examples/yann-lecun_resize.jpg",
        "examples/schmidhuber_resize.png",
    ]

    model = InstantIDModelPipe(xformers=False)
    model._initalize_pipeline(pretrained_model_name_or_path="checkpoints/sdxl/crystalClearXLArtpad_v10.safetensors")
    grid = model.generate_subjects_image_grid(
        subject_image_path_list=subject_paths,
        pose_image_path=None,
        prompt="A portrait of a person in a clown costume",
        negative_prompt="",
        num_steps=15,
        guidance_scale=5,
        identitynet_strength_ratio=0.8,
        adapter_strength_ratio=0.8,
        pose_strength=0,
        canny_strength=0,
        depth_strength=0,
        controlnet_selection=[]
        )
    grid.save("examples/grid_clown.jpg")


def test_inference_with_lora():
    lora_array = [
        {
        "path" : "checkpoints/LORAs/simple_drawing_xl_b1-000012.safetensors",
        "prompt" : "Simple Drawing",
        "scale" : 1
        
    },
    {
        "path" : "checkpoints/LORAs/122359_detail_tweaker.safetensors",
        "prompt" : "<lora:add-detail-xl:2>",
        "scale" : 1
            }
    ]
    assert os.path.exists("examples/musk_resize.jpeg")
    model = InstantIDModelPipe(xformers=False)
    model._initalize_pipeline(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0")
    model._configure_loras(lora_array)
    image, cropped_face = model._generate_single_image(
        face_image_path="examples/musk_resize.jpeg",
        pose_image_path=None,
        prompt="A line drawing of a clown",
        negative_prompt="",
        num_steps=15,
        guidance_scale=5,
        identitynet_strength_ratio=0.5,
        adapter_strength_ratio=0.5,
        pose_strength=0,
        canny_strength=0,
        depth_strength=0,
        controlnet_selection=[],
    )
    assert image is not None
    # save the image
    image.save("examples/musk_clown_lora.jpg")



def test_double_image_generation():
    # assert that te image exists
    assert os.path.exists("examples/istockphoto-1368004438-612x612.jpg")
    model = InstantIDModelPipe(xformers=False)
    model._initalize_pipeline(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0")
    image, cropped_face = model._generate_with_n_faces(
        face_image_path="examples/istockphoto-1368004438-612x612.jpg",
        n_faces= 2,
        pose_image_path=None,
        prompt="A happy couple in clown costumes",
        negative_prompt="",
        num_steps=15,
        guidance_scale=5,
        identitynet_strength_ratio=0.5,
        adapter_strength_ratio=0.5,
        pose_strength=0,
        canny_strength=0,
        depth_strength=0,
        controlnet_selection=[],
        enhance_face_region=True
    )
    assert image is not None
    # save the image
    image.save("examples/couple_clowns.jpg")

if __name__ == "__main__":
    #test_single_image_generation()
    test_inference_with_lora()
