from __future__ import annotations
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


from adapter.attention_processor import AttnProcessor2_0,MaskedIPAttnProcessor2_0
from adapter.projection import ImageProjModel,Resampler
from adapter.model import MSAdapter
from adapter.utils import get_phrase_idx, get_eot_idx

from engine.unet_v2 import UNet2DConditionModel
from engine.unet_controller import UNetController
from engine.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from engine.utils import get_max_window_length, circular_sliding_windows


def get_phrases_idx(tokenizer, phrases, prompt):
    res = []
    phrase_cnt = {}
    for phrase in phrases:
        if phrase in phrase_cnt:
            cur_cnt = phrase_cnt[phrase]
            phrase_cnt[phrase] += 1
        else:
            cur_cnt = 0
            phrase_cnt[phrase] = 1
        res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
    return res

@dataclass
class StoryTailorConfig:
    base_model_path: str = "models/stable-diffusion-xl-base-1.0"
    vae_model_path: Optional[str] = None
    image_encoder_path: str = "models/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    ms_adapter_ckpt: str = "models/MS-Diffusion/ms_adapter_1.bin"

    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    variant: str = "fp16"

    image_proj_type: str = "resampler"   # linear | resampler
    latent_init_mode: str = "grounding"

    num_tokens: int = 16
    text_tokens: Optional[int] = None
    mixed_precision: str = "fp16"        # fp16 | bf16 | fp32


class StoryTailor:
    """
    按 inference4.py 封装的最小可用 StoryTailor 类
    """

    def __init__(self, cfg: StoryTailorConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        self.image_processor = CLIPImageProcessor()

        self.pipe = self._build_pipeline()
        self.image_encoder = self._build_image_encoder()
        self.image_proj_model = self._build_image_proj_model()
        self.unet_controller = self._build_unet_controller()
        self.ms_adapter = self._build_ms_adapter()

    # =========================
    # build
    # =========================
    def _build_pipeline(self) -> StableDiffusionXLPipeline:
        base = self.cfg.base_model_path
        vae_path = self.cfg.vae_model_path or base

        scheduler = EulerDiscreteScheduler.from_pretrained(
            base,
            subfolder="scheduler",
            torch_dtype=self.dtype,
            variant=self.cfg.variant,
        )

        tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")

        text_encoder = CLIPTextModel.from_pretrained(
            base,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        ).to(self.device)

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
        ).to(self.device)

        unet = UNet2DConditionModel.from_pretrained(
            base,
            subfolder="unet",
            torch_dtype=self.dtype,
            variant=self.cfg.variant,
            low_cpu_mem_usage=False,
        ).to(self.device)

        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=self.dtype,
        ).to(self.device)

        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=False)

        self.text_tokens = self.cfg.text_tokens or text_encoder.config.max_position_embeddings
        return pipe


    def _build_image_encoder(self) -> CLIPVisionModelWithProjection:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.image_encoder_path
        ).to(self.device, dtype=self.dtype)
        image_encoder.requires_grad_(False)
        return image_encoder

    def _build_image_proj_model(self):
        cross_attention_dim = self.pipe.unet.config.cross_attention_dim

        if self.cfg.image_proj_type == "linear":
            model = ImageProjModel(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=self.cfg.num_tokens,
            ).to(self.device, dtype=self.dtype)

        elif self.cfg.image_proj_type == "resampler":
            model = Resampler(
                dim=1280,
                depth=4,
                dim_head=64,
                heads=20,
                num_queries=self.cfg.num_tokens,
                embedding_dim=self.image_encoder.config.hidden_size,
                output_dim=cross_attention_dim,
                ff_mult=4,
                latent_init_mode=self.cfg.latent_init_mode,
                phrase_embeddings_dim=self.pipe.text_encoder.config.projection_dim,
            ).to(self.device, dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported image_proj_type: {self.cfg.image_proj_type}")

        return model

    def _build_unet_controller(self) -> UNetController:
        controller = UNetController()
        controller.device = str(self.device)
        controller.tokenizer = self.pipe.tokenizer
        return controller

    def _build_ms_adapter(self) -> MSAdapter:
        """
        对齐 inference4.py 的 attention processor 构造逻辑
        """
        unet = self.pipe.unet
        attn_procs = {}
        unet_sd = unet.state_dict()

        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                continue

            if cross_attention_dim is None or name.endswith("attn1"):
                attn_procs[name] = AttnProcessor2_0()
            else:
                if name.endswith("attn2"):
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    }

                    attn = MaskedIPAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.cfg.num_tokens,
                        text_tokens=self.text_tokens,
                        scale=1.0,
                        need_text_attention_map=False,
                        need_image_attention_map=True,
                    )
                    attn.layer_name = layer_name
                    attn.load_state_dict(weights)
                    attn_procs[name] = attn
                    setattr(attn_procs[name], "__tag__", layer_name)

        unet.set_attn_processor(attn_procs)

        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values()).to(
            self.device, self.dtype
        )

        return MSAdapter(
            unet=unet,
            image_proj_model=self.image_proj_model,
            adapter_modules=adapter_modules,
            ckpt_path=self.cfg.ms_adapter_ckpt,
            num_tokens=self.cfg.num_tokens,
            text_tokens=self.text_tokens,
            device=str(self.device),
        )
    # =========================
    # utils
    # =========================
    def _get_weight_dtype(self):
        if self.cfg.mixed_precision == "fp16":
            return torch.float16
        if self.cfg.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def prepare_reference_images(self, reference_images: Sequence[Image.Image]) -> torch.Tensor:
        images = [img.convert("RGB").resize((512, 512)) for img in reference_images]
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values
        return pixel_values.unsqueeze(0).to(self.device, dtype=self.dtype)

    def _normalize_phrases(
        self,
        phrases: Optional[Union[Sequence[Sequence[str]], Sequence[str]]],
        num_refs: int,
    ):
        if phrases is None:
            return [[f"subject_{i}" for i in range(num_refs)]]
        if len(phrases) > 0 and isinstance(phrases[0], str):
            return [list(phrases)]
        return phrases

    # =========================
    # generation
    # =========================

    def generate_story(
        self,
        id_prompt: str,
        frame_prompt_list: Sequence[str],
        reference_images: Sequence[Image.Image],
        boxes: List[List[List[float]]],
        result_path: str,
        phrases: Optional[Union[Sequence[Sequence[str]], Sequence[str]]] = None,
        negative_prompt: str = "",
        action_list: Optional[Sequence[Optional[str]]] = None,
        seed: int = 42,
        num_samples: int = 1,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        tau: float = 0.85,
        verbose: bool = False,
    ):
        windows_length = 10
        unet_controller = self.unet_controller
        pipe=self.pipe
        self.image_proj_model = self._build_image_proj_model()
        image_encoder=self.image_encoder
        input_images = reference_images
        device = self.device
        image_processor = self.image_processor
        ms_adapter=self.ms_adapter
        drop_grounding_tokens = [0]
        timestamp = int(time.time())  # 为本次生成的所有帧创建一个统一的时间戳
        subfolder_path = os.path.join(result_path, str(timestamp))  # 创建统一的子文件夹

        self.unet_controller.tau = tau

        max_window_length = get_max_window_length(unet_controller, id_prompt, frame_prompt_list)
        window_length = min(windows_length, max_window_length)
        if window_length < len(frame_prompt_list):
            movement_lists = circular_sliding_windows(frame_prompt_list, window_length)
        else:
            movement_lists = [movement for movement in frame_prompt_list]
        story_images = []

        for index, movement in enumerate(frame_prompt_list):
            action_prompt = [f'{action_list[index]}']
            if unet_controller is not None:
                if window_length < len(frame_prompt_list):
                    unet_controller.frame_prompt_suppress = movement_lists[index][1:]
                    unet_controller.frame_prompt_express = movement_lists[index][0]
                    gen_propmts = [f'{id_prompt} {" ".join(movement_lists[index])}']
                    action_prompt = [f'{action_list[index]}']

                else:
                    unet_controller.frame_prompt_suppress = movement_lists[:index] + movement_lists[index + 1:]
                    unet_controller.frame_prompt_express = movement_lists[index]
                    gen_propmts = [f'{id_prompt} {" ".join(movement_lists)}']
                    action_prompt = [f'{action_list[index]}']

                if verbose:
                    print(f"suppress: {unet_controller.frame_prompt_suppress}")
                    print(f"express: {unet_controller.frame_prompt_express}")
                    print(f'id_prompt: {id_prompt}')
                    print(f"gen_propmts: {gen_propmts}")
                    print(f"action_prompts:{action_prompt}")
            else:
                gen_propmts = f'{id_prompt} {movement}'

            guidance_scale = 1.4- tau
            phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], gen_propmts[0])]
            eot_idxes = [[get_eot_idx(pipe.tokenizer, gen_propmts[0])] * len(phrases[0])]

            images = ms_adapter.generate(
                pipe=pipe,
                pil_images=[input_images],
                num_samples=num_samples,
                device=device,
                num_inference_steps=num_inference_steps,
                seed=seed,
                prompt=[gen_propmts[0]],
                negative_prompt=[negative_prompt],
                scale=guidance_scale,
                image_encoder=image_encoder,
                image_processor=image_processor,
                boxes=boxes,
                image_proj_type="resampler",
                image_encoder_type="clip",
                phrases=phrases,
                drop_grounding_tokens=drop_grounding_tokens,
                phrase_idxes=phrase_idxes,
                eot_idxes=eot_idxes,
                height=height,
                width=width,
                action=action_prompt,
                unet_controller=unet_controller,

            )


            images = images[0]
            story_images.append(images)
            os.makedirs(subfolder_path, exist_ok=True)
            images.save(os.path.join(subfolder_path, f'{id_prompt} {unet_controller.frame_prompt_express}.jpg'))

            image_array_list = [np.array(pil_img) for pil_img in story_images]

            frame_fname = f"{id_prompt} {unet_controller.frame_prompt_express}.jpg"
            frame_path = os.path.join(subfolder_path, frame_fname)

            # Concatenate images horizontally
            story_image = np.concatenate(image_array_list, axis=1)
            story_image = Image.fromarray(story_image.astype(np.uint8))

            if unet_controller.Save_story_image:
                story_image.save(os.path.join(subfolder_path, f'story_image_{id_prompt}.jpg'))
