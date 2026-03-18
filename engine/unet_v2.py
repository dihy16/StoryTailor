# modified from the https://github.com/cloneofsimo/minSDXL

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from typing import Optional, Dict

from pygments.lexer import inherit
from .utils import *
from engine.unet_controller import UNetController

# SDXL


class Timesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class Attention(nn.Module):
    def __init__(
            self, inner_dim, cross_attention_dim=None, num_heads=None, dropout=0.0
    ):
        super(Attention, self).__init__()
        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim ** -0.5
        if cross_attention_dim is None:
            cross_attention_dim = inner_dim
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim), nn.Dropout(dropout, inplace=False)]
        )

        # Compatibility for AttnProcessor2_0
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.norm_encoder_hidden_states = nn.LayerNorm(cross_attention_dim)
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.heads = self.num_heads

        self.processor = None

    def set_processor(self, processor:nn.Module):
        self.processor = processor

    def set_use_memory_efficient_attention_xformers(self, use_xformers: bool):
        if hasattr(self.processor, "set_use_memory_efficient_attention_xformers"):
            self.processor.set_use_memory_efficient_attention_xformers(use_xformers)

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def get_raw_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = torch.float16


        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        baddbmm_input=baddbmm_input.to(dtype)

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        # if self.upcast_softmax:
        #     attention_scores = attention_scores.float()

        return attention_scores
    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = torch.float16


        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        baddbmm_input=baddbmm_input.to(dtype)

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        # if self.upcast_softmax:
        #     attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def forward(self, hidden_states, encoder_hidden_states=None,unet_controller: Optional[UNetController] = None, cross_attention_kwargs=None,):
        if  self.processor is not None:
            if encoder_hidden_states is not None:
                attn_output = self.processor(self, hidden_states=hidden_states,
                                             encoder_hidden_states=encoder_hidden_states,
                                             cross_attention_kwargs=cross_attention_kwargs,
                                              unet_controller=unet_controller)
            else:
                attn_output = self.processor(self, hidden_states=hidden_states,
                                             encoder_hidden_states=encoder_hidden_states,
                                             cross_attention_kwargs=cross_attention_kwargs,
                                             unet_controller=unet_controller
                                             )
            return attn_output

        q = self.to_q(hidden_states)
        k = (
            self.to_k(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_k(hidden_states)
        )
        v = (
            self.to_v(encoder_hidden_states)
            if encoder_hidden_states is not None
            else self.to_v(hidden_states)
        )
        b, t, c = q.size()

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if (
                unet_controller is not None and unet_controller.Use_ipca and unet_controller.current_unet_position in unet_controller.Ipca_position
                and encoder_hidden_states is not None and unet_controller.current_time_step >= unet_controller.Ipca_start_step):

            if unet_controller.do_classifier_free_guidance is True:
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = torch.softmax(scores, dim=-1)  # this is only used by cross_attn_map store
                ipca_attn_output = ipca2(q, k, v, self.scale, unet_controller=unet_controller)
                attn_output = ipca_attn_output
            else:
                exit("current doesn't support cfg=1.0")

        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)

        for layer in self.to_out:
            attn_output = layer(attn_output)

        return attn_output


class GEGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=True)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.nn.functional.gelu(x2)


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeedForward, self).__init__()

        self.net = nn.ModuleList(
            [
                GEGLU(in_features, out_features * 4),
                nn.Dropout(p=0.0, inplace=False),
                nn.Linear(out_features * 4, out_features, bias=True),
            ]
        )

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn1 = Attention(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.attn2 = Attention(hidden_size, 2048)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-05, elementwise_affine=True)
        self.ff = FeedForward(hidden_size, hidden_size)

    def forward(self, x, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None,cross_attention_kwargs=None):
        residual = x

        x = self.norm1(x)
        x = self.attn1(x, unet_controller=unet_controller)
        x = x + residual

        residual = x

        x = self.norm2(x)
        if encoder_hidden_states is not None:
            x = self.attn2(x, encoder_hidden_states, unet_controller=unet_controller,cross_attention_kwargs = cross_attention_kwargs)
        else:
            x = self.attn2(x, unet_controller=unet_controller,cross_attention_kwargs = cross_attention_kwargs)
        x = x + residual

        residual = x

        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        return x


class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(Transformer2DModel, self).__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(in_channels, out_channels, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(out_channels) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None,cross_attention_kwargs=None):
        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * width, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, unet_controller=unet_controller,cross_attention_kwargs = cross_attention_kwargs)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=False),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states, temb)
            output_states.append(hidden_states)

        hidden_states = self.downsamplers[0](hidden_states)
        output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True):
        super(CrossAttnDownBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.downsamplers = None
        if has_downsamplers:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, out_channels)]
            )

    def forward(self, hidden_states, temb, encoder_hidden_states, unet_controller: Optional[UNetController] = None , cross_attention_kwargs=None):
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                unet_controller=unet_controller,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers):
        super(CrossAttnUpBlock2D, self).__init__()
        self.attentions = nn.ModuleList(
            [
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
                Transformer2DModel(out_channels, out_channels, n_layers),
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(prev_output_channel + out_channels, out_channels),
                ResnetBlock2D(2 * out_channels, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, unet_controller: Optional[UNetController] = None,cross_attention_kwargs = None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if unet_controller is not None and unet_controller.Is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    0 if unet_controller.current_unet_position == 'up0' else 1,
                    hidden_states,
                    res_hidden_states,
                    s1=unet_controller.Freeu_parm['s1'],
                    s2=unet_controller.Freeu_parm['s2'],
                    b1=unet_controller.Freeu_parm['b1'],
                    b2=unet_controller.Freeu_parm['b2'],
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                unet_controller=unet_controller,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels),
                ResnetBlock2D(out_channels * 2, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, in_features):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.attentions = nn.ModuleList(
            [Transformer2DModel(in_features, in_features, n_layers=10)]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, unet_controller: Optional[UNetController] = None,cross_attention_kwargs = None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                unet_controller=unet_controller,
                cross_attention_kwargs = cross_attention_kwargs,
            )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNet2DConditionModel(ModelMixin, ConfigMixin):
    def __init__(self):
        super(UNet2DConditionModel, self).__init__()  ## init child class first

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        # self.config = namedtuple(
        #             "config", "in_channels addition_time_embed_dim sample_size time_cond_proj_dim"
        #         )
        # self.config.in_channels = 4
        # self.config.addition_time_embed_dim = 256
        # self.config.sample_size = 128
        # self.config.time_cond_proj_dim = None
        self.attn_processors = {}

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)
        self.time_proj = Timesteps()
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280)
        self.add_time_proj = Timesteps(256)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_channels=320, out_channels=320),
                CrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2),
                CrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,

                ),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                CrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640),
            ]
        )
        self.mid_block = UNetMidBlock2DCrossAttn(1280)
        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1)
        self.register_attention_modules()


    def set_attn_processor(self, processor_dict: Dict[str, nn.Module]):
        for name, processor in processor_dict.items():
            module = self
            parts = name.split(".")
            for part in parts[:-1]:
                module = getattr(module, part)
            attn_module = getattr(module, parts[-1])
            if hasattr(attn_module, "set_processor"):
                setattr(processor, "__tag__", name)
                if hasattr(attn_module, "heads"):
                    setattr(processor, "heads", attn_module.heads)
                attn_module.set_processor(processor)
                self.attn_processors[name] = processor


    def get_attn_processors(self):
        return self.attn_processors

    def register_attention_modules(self):
        self.attn_processors = {}
        for name, module in self.named_modules():
            if isinstance(module, Attention):
                self.attn_processors[name] = module.processor  # None by default


    @property
    def attention_processors(self):
        return self.attn_processors

    def forward(
            self, sample, timesteps, encoder_hidden_states, cross_attention_kwargs, added_cond_kwargs,
            unet_controller: Optional[UNetController] = None, **kwargs
    ):
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        if cross_attention_kwargs is not None:
            boxes = cross_attention_kwargs.get("boxes")
            phrase_idxes = cross_attention_kwargs.get("phrase_idxes")
            eot_idxes = cross_attention_kwargs.get("eot_idxes")
            inherit_rect = cross_attention_kwargs.get("inherit_rect")
            action_embeds = cross_attention_kwargs.get("action_embeds")
            action_mask = cross_attention_kwargs.get("action_mask")
            action_map_save_dir = cross_attention_kwargs.get("action_map_save_dir")
            action_map_upsample_to = cross_attention_kwargs.get("action_map_upsample_to")
            action_map_use_focus_heads = cross_attention_kwargs.get("action_map_use_focus_heads")
            cross_attention_kwargs = {"boxes": boxes,
                                      "phrase_idxes": phrase_idxes,
                                      "eot_idxes": eot_idxes,
                                      "inherit_rect": inherit_rect,
                                      "action_embeds": action_embeds,
                                      "action_mask": action_mask,
                                      "action_map_save_dir": action_map_save_dir,
                                      "action_map_upsample_to":action_map_upsample_to,
                                      "action_map_use_focus_heads": action_map_use_focus_heads
                                      }


        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb

        sample = self.conv_in(sample)

        # 3. down
        if unet_controller is not None:
            unet_controller.current_unet_position = 'down0'

        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'down1'

        # encoder_hidden_states is prompt_embedings, so here do cross_attn
        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,  # time_embbeding
            encoder_hidden_states=encoder_hidden_states,
            # [2,77,2048], 2 means two branch, 1 for prompt, 1 for negative prompt
            unet_controller=unet_controller,
            cross_attention_kwargs = cross_attention_kwargs,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'down2'

        sample, [s7, s8] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            unet_controller=unet_controller,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        # 4. mid
        if unet_controller is not None:
            unet_controller.current_unet_position = 'mid'

        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, unet_controller=unet_controller,cross_attention_kwargs = cross_attention_kwargs,
        )

        # 5. up
        if unet_controller is not None:
            unet_controller.current_unet_position = 'up0'

        sample = self.up_blocks[0](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
            unet_controller=unet_controller,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'up1'

        sample = self.up_blocks[1](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
            unet_controller=unet_controller,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        if unet_controller is not None:
            unet_controller.current_unet_position = 'up2'

        sample = self.up_blocks[2](
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return [sample]