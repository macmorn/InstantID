import os
import sys

sys.path.append("./")
from gird_search_images import InstantIDModelPipe


def test_single_image_generation():
    # assert that te image exists
    assert os.path.exists("examples/musk_resize.jpeg")
    model = InstantIDModelPipe(xformers=False)
    model._initalize_pipeline(pretrained_model_name_or_path="wangqixun/YamerMIX_v8")
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
    model._initalize_pipeline(pretrained_model_name_or_path="wangqixun/YamerMIX_v8")
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


if __name__ == "__main__":
    test_grid_creation()
