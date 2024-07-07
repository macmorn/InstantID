from gird_search_images import InstantIDModelPipe


def test_single_image_generation():
    model = InstantIDModelPipe(xformers=True)
    model._initalize_pipeline(pretrained_model_name_or_path="wangqixun/YamerMIX_v8")
    image = model._generate_single_image(
        face_image_path="examples/musk_resize.jpeg",
        pose_image_path=None,
        prompt="A portrait of a person in a clown costume",
        negative_prompt="",
        num_steps=30,
        identitynet_strength_ratio=0.5,
        adapter_strength_ratio=0.5,
        controlnet_strength_ratio=0.5,
        pose_strength=0.5,
        canny_strength=0,
        depth_strength=0,
        controlnet_selection=[],
        guidance_scale=5,
    )
    assert image is not None
    # save the image
    image.save("examples/musk_clown.jpg")
