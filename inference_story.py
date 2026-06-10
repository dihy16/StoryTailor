from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from storytailor import StoryTailor, StoryTailorConfig
import diffusers

diffusers.utils.logging.set_verbosity_error()

def main():

    base_model_path = 'models/stable-diffusion-xl-base-1.0'
    image_encoder_path = 'models/CLIP-ViT-bigG-14-laion2B-39B-b160k'
    ms_adapter_ckpt = 'models/MS-Diffusion/ms_adapter.bin'


    image1 = Image.open('./examples/example_cat.jpg')
    image2 = Image.open('./examples/example_dog.jpg')

    input_images = [image2, image1]
    input_images = [img.convert('RGB').resize((512, 512)) for img in input_images]

    boxes = [[[0.25, 0.2, 0.6, 0.6], [0.5, 0.20, 0.85, 0.50]]]  # dog+cat
    phrases = [['dog', 'cat']]

    id_prompt = 'best quality, high quality, the dog and the cat'
    frame_prompt_list = [
        'are dancing and holding hands.',
        'are fighting with each other, bed.',
        'are hugging with each other.',
        'are nestling in living room.',
    ]
    negative_prompt = 'low quality, worst quality, bad anatomy, red color'

    action_list = [
        'dancing.',
        'fighting.',
        'hugging.',
        'nestling.',
    ]

    # =========================
    # 3. generation config
    # =========================
    num_samples = 1
    height = 1024
    width = 1024
    num_inference_steps = 30
    seed = 7
    tau = 0.85
    verbose = True

    output_dir = 'outputs/storytailor_story'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # =========================
    # 4. 初始化 StoryTailor
    # =========================
    cfg = StoryTailorConfig(
        base_model_path=base_model_path,
        image_encoder_path=image_encoder_path,
        ms_adapter_ckpt=ms_adapter_ckpt,
    )
    model = StoryTailor(cfg)

    # =========================
    # 5. 多帧推理
    # =========================
    all_frame_outputs = model.generate_story(
        id_prompt=id_prompt,
        frame_prompt_list=frame_prompt_list,
        reference_images=input_images,
        boxes=boxes,
        result_path=output_dir,
        phrases=phrases,
        negative_prompt=negative_prompt,
        action_list=action_list,
        seed=seed,
        num_samples=num_samples,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        tau = tau,
        verbose=verbose,
    )



if __name__ == '__main__':
    main()
