# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
# and https://github.com/MS-Diffusion/MS-Diffusion/blob/main/msdiffusion/models/attention_processor.py
import os
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.unet_controller import UNetController


def minmax_normalize(batch_maps):
    min_val = batch_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = batch_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    return (batch_maps - min_val) / (max_val - min_val + 1e-5)

def _to_vis_prob(prob: torch.Tensor) -> torch.Tensor:
    """
    将任意 attention 概率张量整理成可视化用的 vis_prob：
    1) detatch + float32
    2) 去 NaN/Inf
    3) 按最后一维做 [0,1] 归一化，避免全黑/全白
    形状保持不变（通常是 [B*H, Q, K] 或 [B, H, Q, K]）
    """
    prob = prob.detach().to(torch.float32)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = prob.amax(dim=-1, keepdim=True)
    prob = prob / (maxv + 1e-8)
    return prob



class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            boxes=None,
            phrase_idxes=None,
            eot_idxes=None,
            cross_attention_kwargs=None,
            unet_controller: Optional[UNetController] = None
    ):
        residual = hidden_states


        if attn.spatial_norm is not None:
            print(" spatial this step have done")
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            print(" normal this step have done")
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        #新增片段
        controller = unet_controller
        # controller = None
        use_kv_cache = (encoder_hidden_states is not None) and (controller is not None)
        if use_kv_cache:
            q_bh = query.reshape(batch_size, attn.heads, -1, head_dim)
            k_bh = key.reshape(batch_size, attn.heads, -1, head_dim)
            v_bh = value.reshape(batch_size, attn.heads, -1, head_dim)
            mask_bhq = None
            if attention_mask is not None:
                mask_bhq = attention_mask.reshape(batch_size * attn.heads, attention_mask.shape[-2], attention_mask.shape[-1]).to(q_bh.dtype)
            scale = getattr(attn,"scale",1.0 / math.sqrt(head_dim))
            out_bh = controller.xattn_apply_kv_cache(
                q_bh, k_bh, v_bh, scale,
                time_step=controller.current_unet_position,
                unet_position=controller.current_unet_position,
                attn_index=0,
                source_tag="text",
                do_uncond=controller.do_classifier_free_guidance,
                batch_index=None,
                attention_mask=mask_bhq,
                hist_topk=64,  # 取最相关 64 个历史键
                hist_score_bias=0.35,  # 轻微偏向历史（0.1~0.25）
            )
            hidden_states = out_bh.view(batch_size, attn.heads, -1, head_dim)
            hidden_states = hidden_states.transpose(1,2).reshape(batch_size,-1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)


        #新增片段

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        #
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MaskedIPAttnProcessor2_0(nn.Module):

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, text_tokens=77,
                 need_text_attention_map=False, need_image_attention_map=True, num_dummy_tokens=4, mask_threshold=0.5,
                 use_psuedo_attention_mask=True, subject_scales=None, start_step=5):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.text_tokens = text_tokens
        self.num_dummy_tokens = num_dummy_tokens
        self.mask_threshold = mask_threshold
        self.subject_scales = subject_scales
        self.start_step = start_step

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.need_text_attention_map = need_text_attention_map
        self.need_image_attention_map = need_image_attention_map

        self.use_psuedo_attention_mask = use_psuedo_attention_mask
        self.attention_maps = []
        self.image_attention_maps = []
        self.strict_bg_gate = False
        self.vis_prob=None

        self.intervene_logits = True
        self.lambda_map = 2.0
        self.map_threshold = 0.60
        self.bias_clip = 2.5
        self.dynamic_sigma = True
        self.sigma_min = 0.6
        self.sigma_max = 1.6

        self.need_action_attention_map = False
        self.action_attention_maps = []
        self.action_map_save_dir = None
        self.action_map_upsample_to = None
        self.action_map_reduce_heads = "mean"
        self.action_map_reduce_tokens = "mean"

    def _gate_bg_probs_keep_dummy(self,ip_attention_probs, post_psuedo_dummy_attention_mask, num_dummy_tokens):
        if post_psuedo_dummy_attention_mask is None or num_dummy_tokens <= 0:
            return ip_attention_probs

        if ip_attention_probs.dim() == 4:
            B, H, Q, K = ip_attention_probs.shape
            probs = ip_attention_probs.reshape(B * H, Q, K)
        else:
            BH, Q, K = ip_attention_probs.shape
            probs = ip_attention_probs

        bg_mask = post_psuedo_dummy_attention_mask  # [BH, Q]，bool

        device = probs.device
        dtype = probs.dtype
        gate_dummy = torch.zeros(1, 1, K, device=device, dtype=dtype)  # [1,1,K]
        gate_dummy[..., :num_dummy_tokens] = 1.0
        gate_full = torch.ones_like(gate_dummy)
        gate = torch.where(bg_mask.unsqueeze(-1), gate_dummy, gate_full)  # [BH,Q,K]

        probs = probs * gate
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs / denom

        if ip_attention_probs.dim() == 4:
            probs = probs.reshape(B, H, Q, K)
        return probs

    def prepare_attention_mask_qk_two_stage_gaussian_new(
            self,
            boxes,
            phrase_idxes,
            sequence_length_q,
            sequence_length_k,
            batch_size,
            head_size,
            dtype,
            device,
            use_masked_text_attention=False,
            attn_centers_override: Optional[torch.Tensor] = None,
    ):

        if boxes is None:
            return None, None, None

        B, N = boxes.shape[:2]
        P = sequence_length_q
        patch_res = int(P ** 0.5)
        if patch_res * patch_res != P:
            raise ValueError(f"sequence_length_q={P} is not square.")

        ys, xs = torch.meshgrid(
            torch.arange(patch_res, device=device),
            torch.arange(patch_res, device=device),
            indexing="ij",
        )
        patch_coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2).to(torch.float32)  # [P, 2]
        patch_x = patch_coords[:, 0]
        patch_y = patch_coords[:, 1]

        # --------------------------------------------------
        # 1) init box masks
        # --------------------------------------------------
        box_masks = torch.zeros((B, N, P), dtype=torch.bool, device=device)

        for b in range(B):
            for n in range(N):
                x1 = int(torch.floor(boxes[b, n, 0] * patch_res).item())
                y1 = int(torch.floor(boxes[b, n, 1] * patch_res).item())
                x2 = int(torch.ceil(boxes[b, n, 2] * patch_res).item())
                y2 = int(torch.ceil(boxes[b, n, 3] * patch_res).item())

                x1 = max(0, min(patch_res - 1, x1))
                y1 = max(0, min(patch_res - 1, y1))
                x2 = max(0, min(patch_res, x2))
                y2 = max(0, min(patch_res, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                idxs = (patch_x >= x1) & (patch_x < x2) & (patch_y >= y1) & (patch_y < y2)
                box_masks[b, n, idxs] = True

        # --------------------------------------------------
        # 2) XOR：
        #    sum==1 keep
        #    sum>=2 clean
        # --------------------------------------------------
        cover_count = box_masks.sum(dim=1)  # [B, P]
        exclusive_patch = (cover_count == 1)
        overlap_patch = (cover_count >= 2)
        xor_box_masks = box_masks & exclusive_patch.unsqueeze(1)  # [B, N, P]

        r1_base, r2_base = 0.45, 0.7
        sigma_scales = getattr(self, "_sigma_scales", None)

        final_weights = torch.zeros((B, P, N), dtype=torch.float32, device=device)

        for b in range(B):
            for n in range(N):
                mask_bn = xor_box_masks[b, n]
                if not bool(mask_bn.any()):
                    continue

                x1 = int(torch.floor(boxes[b, n, 0] * patch_res).item())
                y1 = int(torch.floor(boxes[b, n, 1] * patch_res).item())
                x2 = int(torch.ceil(boxes[b, n, 2] * patch_res).item())
                y2 = int(torch.ceil(boxes[b, n, 3] * patch_res).item())

                x1 = max(0, min(patch_res - 1, x1))
                y1 = max(0, min(patch_res - 1, y1))
                x2 = max(0, min(patch_res, x2))
                y2 = max(0, min(patch_res, y2))

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                scale = 1.0
                if self.dynamic_sigma and sigma_scales is not None:
                    scale = float(sigma_scales[b, n])

                r1_ratio = max(0.05, min(0.60, r1_base * scale))
                r2_ratio = max(r1_ratio + 1e-4, min(0.75, r2_base * scale))

                if attn_centers_override is not None:
                    center = attn_centers_override[b, n].to(torch.float32)
                else:
                    center = torch.tensor(
                        [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                        device=device,
                        dtype=torch.float32,
                    )

                box_size = max(w, h)
                r1 = max(1.0, r1_ratio * box_size)
                r2 = max(r1 + 1.0, r2_ratio * box_size)

                sigma1 = max(1.5, r1 / 3.0)
                sigma2 = max(1.5, r2 / 6.0)

                idxs_box = torch.nonzero(mask_bn, as_tuple=False).squeeze(1)
                coords_box = patch_coords[idxs_box]
                d = torch.norm(coords_box - center[None, :], dim=1)

                inner = torch.exp(- (d / sigma1) ** 2)
                outer = torch.exp(- ((d - r1).clamp_min(0.0) / sigma2) ** 2)

                decay = torch.where(
                    d <= r1,
                    inner,
                    torch.where(d <= r2, 0.35 * outer, torch.zeros_like(d))
                )

                decay = decay.clamp(0.0, 1.0)
                final_weights[b, idxs_box, n] = decay

        # --------------------------------------------------
        # 4) image bias
        # --------------------------------------------------
        final_weights = final_weights.to(dtype)
        final_weights = final_weights.repeat_interleave(self.num_tokens, dim=-1)  # [B,P,N*T]
        attention_mask_qk_image = (1.0 - final_weights) * -10000.0

        # --------------------------------------------------
        # 5) dummy/background
        # --------------------------------------------------
        covered_after_xor = xor_box_masks.any(dim=1)  # [B,P]
        dummy_mask = ~covered_after_xor
        dummy_attention_mask = dummy_mask.repeat_interleave(head_size, dim=0)

        dummy_bias = dummy_mask.unsqueeze(-1).repeat_interleave(self.num_dummy_tokens, dim=-1)
        dummy_bias = (1.0 - dummy_bias.to(dtype)) * -10000.0
        attention_mask_qk_image = torch.cat([dummy_bias, attention_mask_qk_image], dim=-1)
        attention_mask_qk_image = attention_mask_qk_image.repeat_interleave(head_size, dim=0)

        # --------------------------------------------------
        # 6) optional text mask
        # --------------------------------------------------
        attention_mask_qk_text = None
        if use_masked_text_attention and phrase_idxes is not None:
            attention_mask_qk_text = torch.full(
                (batch_size, P, sequence_length_k),
                fill_value=-10000.0,
                dtype=dtype,
                device=device,
            )
            for b in range(batch_size):
                for n, (start, end) in enumerate(phrase_idxes[b]):
                    if start == 0 and end == 0:
                        continue
                    attention_mask_qk_text[b, xor_box_masks[b, n], start:end] = 0.0

            attention_mask_qk_text = attention_mask_qk_text.repeat_interleave(head_size, dim=0)

        return attention_mask_qk_image, attention_mask_qk_text, dummy_attention_mask



    def get_text_attention_maps(self, attention_probs, boxes, phrase_idxes, head_size):
        bsz = boxes.shape[0]
        _, num_tokens_q, num_tokens_k = attention_probs.shape
        attention_probs = attention_probs.view(bsz, head_size, num_tokens_q, num_tokens_k)
        num_ref = boxes.shape[1]
        h = w = int(num_tokens_q ** 0.5)
        batch_attention_maps = []
        for i in range(bsz):
            sample_attention_maps = []
            for j in range(num_ref):
                start_idx, end_idx = int(phrase_idxes[i, j, 0].item()), int(phrase_idxes[i, j, 1].item())
                if start_idx == 0 and end_idx == 0:
                    sample_attention_maps.append(
                        torch.zeros(num_tokens_q, dtype=attention_probs.dtype, device=attention_probs.device))
                else:
                    attention_map = attention_probs[i, :, :,
                                    start_idx:end_idx]  # [num_heads, num_tokens_q, num_tokens_phrase]
                    attention_map = torch.mean(torch.mean(attention_map, dim=-1), dim=0)  # [num_tokens_q]
                    sample_attention_maps.append(attention_map)
            batch_attention_maps.append(torch.stack(sample_attention_maps))

        self.attention_maps.append(torch.stack(batch_attention_maps).reshape(bsz, num_ref, h, w))

    def get_psuedo_attention_mask(self, head_size):
        if not self.use_psuedo_attention_mask or len(self.attention_maps) < self.start_step :
            return None, None
        text_attention_maps = torch.stack(self.attention_maps).mean(dim=0)  # [bsz, num_ref, h, w]
        text_attention_maps = minmax_normalize(text_attention_maps)
        dtype, device = text_attention_maps.dtype, text_attention_maps.device
        bsz, num_ref, h, w = text_attention_maps.shape
        seq_len_q = h * w
        text_attention_maps = text_attention_maps.view(bsz, num_ref, -1)
        text_attention_maps = text_attention_maps.transpose(1, 2)  # [bsz, h*w, num_ref]

        # use threshold to get the mask
        psuedo_attention_mask = (text_attention_maps > self.mask_threshold).to(dtype)
        psuedo_dummy_attention_mask = torch.ones((bsz, seq_len_q), dtype=dtype, device=device)
        for i in range(num_ref):
            psuedo_box_mask = psuedo_attention_mask[..., i]
            psuedo_dummy_attention_mask = torch.clamp(psuedo_dummy_attention_mask - psuedo_box_mask, min=0)

        # post mask
        post_psuedo_dummy_attention_mask = psuedo_dummy_attention_mask.to(torch.bool)
        post_psuedo_dummy_attention_mask = post_psuedo_dummy_attention_mask.repeat_interleave(head_size, dim=0)

        psuedo_attention_mask = psuedo_attention_mask.repeat_interleave(self.num_tokens, dim=-1)
        psuedo_attention_mask = (1 - psuedo_attention_mask) * -10000.0  # mask to bias
        psuedo_dummy_attention_mask = psuedo_dummy_attention_mask.unsqueeze(-1).repeat_interleave(self.num_dummy_tokens,
                                                                                                  dim=-1)
        psuedo_dummy_attention_mask = (1 - psuedo_dummy_attention_mask) * -10000.0
        psuedo_attention_mask = torch.cat([psuedo_dummy_attention_mask, psuedo_attention_mask], dim=-1)
        if psuedo_attention_mask.shape[0] < bsz * head_size:
            psuedo_attention_mask = psuedo_attention_mask.repeat_interleave(head_size, dim=0)

        return psuedo_attention_mask, post_psuedo_dummy_attention_mask

    def _compute_ip_centers(
            self,
            ip_attention_probs: torch.Tensor,  # [BH,Q,K] or [B,H,Q,K]
            boxes: torch.Tensor,  # [B,N,4], xyxy in [0,1]
            *, batch_size: int, head_size: int,
            seq_len_q: int, num_dummy_tokens: int, num_tokens_per_subject: int,
            ema: float = 0.8,
    ):
        # reshape -> [B,H,Q,K]
        if ip_attention_probs.dim() == 3:
            BH, Q, K = ip_attention_probs.shape
            B, H = batch_size, head_size
            probs = ip_attention_probs.view(B, H, Q, K)
        else:
            B, H, Q, K = ip_attention_probs.shape
            probs = ip_attention_probs

        h = w = int(seq_len_q ** 0.5)
        N = boxes.shape[1]
        centers = torch.zeros(B, N, 2, device=probs.device, dtype=probs.dtype)

        # precompute grids
        yy, xx = torch.meshgrid(
            torch.arange(h, device=probs.device, dtype=probs.dtype),
            torch.arange(w, device=probs.device, dtype=probs.dtype),
            indexing="ij"
        )
        xx = xx.reshape(-1)  # [Q]
        yy = yy.reshape(-1)

        for b in range(B):
            for n in range(N):
                k0 = num_dummy_tokens + n * num_tokens_per_subject
                k1 = k0 + num_tokens_per_subject
                subj = probs[b, :, :, k0:k1].mean(dim=-1)  # [H,Q]
                heat_q = subj.mean(dim=0)  # [Q]
                heat = heat_q.view(h, w)

                x1 = int(torch.floor(boxes[b, n, 0] * w).item())
                y1 = int(torch.floor(boxes[b, n, 1] * h).item())
                x2 = int(torch.ceil(boxes[b, n, 2] * w).item())
                y2 = int(torch.ceil(boxes[b, n, 3] * h).item())
                x1 = max(0, min(w - 1, x1));
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h - 1, y1));
                y2 = max(0, min(h, y2))

                if x2 <= x1 or y2 <= y1:
                    cx = (boxes[b, n, 0] + boxes[b, n, 2]) * 0.5 * (w - 1)
                    cy = (boxes[b, n, 1] + boxes[b, n, 3]) * 0.5 * (h - 1)
                else:
                    sub = heat[y1:y2, x1:x2]
                    if sub.numel() == 0 or sub.max() <= 1e-8:
                        cx = (boxes[b, n, 0] + boxes[b, n, 2]) * 0.5 * (w - 1)
                        cy = (boxes[b, n, 1] + boxes[b, n, 3]) * 0.5 * (h - 1)
                    else:
                        flat_idx = torch.argmax(sub.reshape(-1))
                        py = flat_idx // sub.shape[1]
                        px = flat_idx % sub.shape[1]
                        cx = x1 + px.to(probs.dtype)
                        cy = y1 + py.to(probs.dtype)

                centers[b, n, 0] = cx  # in patch coords [0..w-1]
                centers[b, n, 1] = cy  # in patch coords [0..h-1]

        # EMA smoothing
        if not hasattr(self, "_ip_centers_ema"):
            self._ip_centers_ema = centers.clone()
        else:
            self._ip_centers_ema = ema * self._ip_centers_ema + (1.0 - ema) * centers
        return self._ip_centers_ema  # [B,N,2] in patch coords

    def _build_map_bias_and_sigma_scales(
            self, *, head_size: int, num_dummy_tokens: int, num_tokens_per_subject: int,
            seq_len_q: int, device, dtype
    ):
        if (not self.intervene_logits) or (not self.use_psuedo_attention_mask):
            return None, None
        if len(self.attention_maps) < self.start_step:
            return None, None

        # 1)  [B,num_ref,H,W] -> [B,P,N]
        text_maps = torch.stack(self.attention_maps).mean(dim=0)  # [B,N,H,W]
        # norm [0,1]
        tmin = text_maps.amin(dim=(2, 3), keepdim=True)
        tmax = text_maps.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        text_maps = (text_maps - tmin) / (tmax - tmin)
        B, N, H, W = text_maps.shape
        P = H * W
        text_maps = text_maps.view(B, N, P).transpose(1, 2).to(dtype=dtype, device=device)  # [B,P,N]

        # 2) map-derived bias：λ * (A - τ)
        A_minus_tau = (text_maps - float(self.map_threshold)).clamp_(-1.0, 1.0)  # [B,P,N]
        # num_tokens
        A_nt = A_minus_tau.repeat_interleave(num_tokens_per_subject, dim=-1)  # [B,P,N*T]
        # dummy no bias
        zeros_dummy = torch.zeros(B, P, num_dummy_tokens, dtype=dtype, device=device)
        map_bias = torch.cat([zeros_dummy, A_nt], dim=-1)  # [B,P,D+N*T]
        map_bias = (self.lambda_map * map_bias).clamp_(-self.bias_clip, self.bias_clip)

        # 3) heads
        map_bias = map_bias.repeat_interleave(head_size, dim=0)  # [B*H,P,D+N*T]
        # reshape to [BH, Q, K]
        map_bias_bh_q_k = map_bias

        # 4)
        cover = (text_maps > float(self.map_threshold)).float().mean(dim=1)  # [B,N]
        sigma_scales = self.sigma_min + (self.sigma_max - self.sigma_min) * cover
        sigma_scales = sigma_scales.clamp(self.sigma_min, self.sigma_max)  # [B,N]

        return map_bias_bh_q_k, sigma_scales


    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            boxes=None,
            phrase_idxes=None,
            eot_idxes=None,
            cross_attention_kwargs=None,
            unet_controller: Optional[UNetController] = None,
    ):
        if unet_controller is not None:
            # print("unet_controller is not None")
            unet_controller = unet_controller


        save_path = "./ip_attention_maps"

        residual = hidden_states
        if boxes is None:
            boxes = cross_attention_kwargs.get("boxes")
            phrase_idxes = cross_attention_kwargs.get("phrase_idxes")
            eot_idxes = cross_attention_kwargs.get("eot_idxes")


        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        rf_attention_mask = None

        custom_attention_masks = self.prepare_attention_mask_qk_two_stage_gaussian_new(boxes, phrase_idxes, hidden_states.shape[1],
                                                                self.text_tokens, batch_size, attn.heads,
                                                                hidden_states.dtype, hidden_states.device,
                                                                use_masked_text_attention=False)


        attention_mask_qk_image, attention_mask_qk_text, dummy_attention_mask = custom_attention_masks
        if attention_mask_qk_image is not None:
            attention_mask_qk_image = attention_mask_qk_image.view(batch_size, attn.heads, -1,
                                                                   attention_mask_qk_image.shape[-1])
        if attention_mask_qk_text is not None:
            attention_mask_qk_text = attention_mask_qk_text.view(batch_size, attn.heads, -1,
                                                                 attention_mask_qk_text.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            # end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            end_pos = self.text_tokens
            # print(encoder_hidden_states.shape)   #(2,113,2048)
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],  # (b,0-77,hw)
                encoder_hidden_states[:, end_pos:, :],  # (b,77-113,hw)
            )
            # print(encoder_hidden_states.shape, ip_hidden_states.shape) #(2,77,2048),(2,36,2048)

            attention_mask, rf_attention_mask = (
                attention_mask[:, :, :, :end_pos],
                attention_mask[:, :, :, end_pos:],
            ) if attention_mask is not None else (None, None)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        action_bias = None

        # strengthen = 0.0
        if unet_controller is not None:
            def tau_to_strength(tau, tau_star=0.85):
                if tau <= tau_star:
                    strengthen = tau / tau_star
                else:
                    strengthen = (1.0 - tau) / (1.0 - tau_star)

                return strengthen

            step_idx = int(getattr(unet_controller, "current_time_step", 0))
            num_inference_steps = int(getattr(unet_controller, "num_inference_steps", 1))
            if num_inference_steps > 1:
                t = step_idx / max(1, num_inference_steps - 1)
                cutoff = 0.4
                if t <= cutoff:
                    strengthen = tau_to_strength(unet_controller.tau)
                else:
                    x = (t - cutoff) / (1.0 - cutoff)
                    strengthen = 0.5 * (1.0 + math.cos(math.pi * x))


        if isinstance(cross_attention_kwargs, dict):
            action_embeds = cross_attention_kwargs.get("action_embeds", None)  # [B,S,C]
            action_mask = cross_attention_kwargs.get("action_mask", None)  # [B,S]，
        else:
            action_embeds = None
            action_mask = None

        if action_embeds is not None:
            B = hidden_states.shape[0]
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            q = query.view(B, -1, attn.heads, head_dim).transpose(1, 2)  # [B,H,Q,hd]
            k_text = key.view(B, -1, attn.heads, head_dim).transpose(1, 2)  # [B,H,K,hd]
            ctx_in = attn.to_k.in_features  # e.g. 640
            ae = action_embeds.to(device=key.device, dtype=key.dtype)
            last = ae.shape[-1]
            if last != ctx_in:
                if last > ctx_in:
                    ae = ae[..., :ctx_in]
                else:
                    pad = ctx_in - last
                    ae = torch.nn.functional.pad(ae, (0, pad), value=0.0)  # [*, last] -> [*, ctx_in]

            akey = attn.to_k(ae)
            k_act = akey.view(B, -1, attn.heads, head_dim).transpose(1, 2)  # [B,H,S,hd]
            k_mean = k_act.mean(dim=2, keepdim=True)  # [B,H,1,hd]
            q_len = hidden_states.shape[1] if hidden_states.dim() == 3 else (height * width)
            scale_lo = 1.0 if q_len <= 256 else 0.5
            h_focus_frac = float(cross_attention_kwargs.get("action_head_frac", 0.5))
            h_focus = max(1, min(attn.heads, int(round(attn.heads * h_focus_frac))))
            gamma = float(cross_attention_kwargs.get("action_q_push", 0.4)) * scale_lo * strengthen
            tau = float(cross_attention_kwargs.get("action_head_temp", 0.7))
            q_slice = q[:, :h_focus, :, :]  # [B,hf,Q,hd]
            q_norm = q_slice.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            q[:, :h_focus, :, :] = torch.nn.functional.normalize(
                q_slice + gamma * k_mean[:, :h_focus, :, :].expand_as(q_slice),
                dim=-1
            ) * q_norm
            q[:, :h_focus, :, :] = q[:, :h_focus, :, :] / tau

            query = q.transpose(1, 2).reshape(B, -1, attn.heads * head_dim)

            if self.need_action_attention_map:
                use_focus_heads = bool(cross_attention_kwargs.get("action_map_use_focus_heads", True))
                q_for_map = q[:, :h_focus] if use_focus_heads else q  # [B,Hm,Q,hd]
                k_for_map = k_act[:, :h_focus] if use_focus_heads else k_act

                scale_attn = 1.0 / math.sqrt(head_dim)
                logits_act = torch.einsum("bhqd,bhsd->bhqs", q_for_map, k_for_map) * scale_attn

                # mask padding action tokens
                if action_mask is not None:
                    am = action_mask.to(device=logits_act.device, dtype=logits_act.dtype)  # [B,S]
                    am4 = am.view(B, 1, 1, -1)  # [B,1,1,S]
                    logits_act = logits_act + (1.0 - am4) * (-1e4)

                prob_act = torch.softmax(logits_act, dim=-1)  # [B,Hm,Q,S]

                reduce_mode = "max"  # "max" / "topk" / "token"
                if isinstance(cross_attention_kwargs, dict):
                    reduce_mode = cross_attention_kwargs.get("action_map_reduce_tokens", reduce_mode)

                if reduce_mode == "token":
                    idx = int(cross_attention_kwargs.get("action_token_idx", 0)) if isinstance(cross_attention_kwargs,
                                                                                               dict) else 0
                    idx = max(0, min(idx, prob_act.shape[-1] - 1))
                    amap_bhq = prob_act[..., idx]  # [B,Hm,Q]
                elif reduce_mode == "topk":
                    k = int(cross_attention_kwargs.get("action_map_topk", 4)) if isinstance(cross_attention_kwargs,
                                                                                            dict) else 4
                    k = max(1, min(k, prob_act.shape[-1]))
                    amap_bhq = prob_act.topk(k, dim=-1).values.mean(dim=-1)  # [B,Hm,Q]
                else:
                    amap_bhq = prob_act.max(dim=-1).values  # [B,Hm,Q]

                # reduce heads
                if self.action_map_reduce_heads == "max":
                    amap_bq = amap_bhq.max(dim=1).values  # [B,Q]
                else:
                    amap_bq = amap_bhq.mean(dim=1)        # [B,Q]

                # reshape Q -> (h,w)
                if input_ndim == 4:
                    Hs, Ws = int(height), int(width)
                else:
                    Qq = int(amap_bq.shape[-1])
                    side = int(math.sqrt(Qq) + 1e-6)
                    Hs = Ws = side if side * side == Qq else Qq  # fallback
                if Hs * Ws == int(amap_bq.shape[-1]):
                    amap_bhw = amap_bq.view(B, Hs, Ws)
                    save_dir = cross_attention_kwargs.get("action_map_save_dir", None) if isinstance(cross_attention_kwargs, dict) else None
                    up_hw = cross_attention_kwargs.get("action_map_upsample_to", None) if isinstance(cross_attention_kwargs, dict) else None
                    self._record_action_map(
                        amap_bhw,
                        attn=attn,
                        unet_controller=unet_controller,
                        save_dir=(save_dir or self.action_map_save_dir),
                        upsample_to=(tuple(up_hw) if up_hw is not None else self.action_map_upsample_to),
                    )

            sim = torch.einsum("bhkd,bhsd->bhks", k_text, k_act) * (1.0 / math.sqrt(head_dim))  # [B,H,K,S]
            if action_mask is not None:
                am4 = action_mask.to(device=sim.device, dtype=sim.dtype).view(B, 1, 1, -1)  # [B,1,1,S]
                sim = sim + (1.0 - am4) * (-1e4)

            bias_k = sim.max(dim=-1).values
            beta = float(cross_attention_kwargs.get("action_bias_scale", 1.6)) * scale_lo * strengthen
            center = bias_k.median(dim=-1, keepdim=True).values
            bias_k = (bias_k - center).relu().tanh() * beta

            Q = query.shape[1];
            K = k_text.shape[2]
            action_bias = bias_k.unsqueeze(2).expand(B, attn.heads, Q, K).to(key.dtype)  # [B,H,Q,K]

        if attention_mask is None:
            attention_mask = action_bias
        else:
            attention_mask = attention_mask + action_bias

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        attention_mask = attention_mask_qk_text if attention_mask_qk_text is not None else attention_mask
        do_uncond = bool(cross_attention_kwargs.get("do_uncond", False)) if (cross_attention_kwargs is not None) else False

        if not self.need_text_attention_map:
            q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ts = getattr(unet_controller, "current_time_step", -1)
            pos = getattr(unet_controller, "current_unet_position", "")
            if not hasattr(attn, "_kv_last_ts") or attn._kv_last_ts != ts or getattr(attn, "_kv_last_pos", None) != pos:
                attn._kv_last_ts = ts;attn._kv_last_pos = pos;attn._kv_call_idx = 0;attn_index = attn._kv_call_idx;attn._kv_call_idx += 1

            use_kv_cache = (unet_controller is not None) and getattr(unet_controller, "CacheMode", "off") != "off"
            if use_kv_cache:
                q_bh = q.reshape(batch_size * attn.heads, -1, head_dim)
                k_bh = k.reshape(batch_size * attn.heads, -1, head_dim)
                v_bh = v.reshape(batch_size * attn.heads, -1, head_dim)
                mask_bhq = None
                if attention_mask is not None:
                    mask_bhq = attention_mask.reshape(batch_size * attn.heads, attention_mask.shape[-2],
                                                      attention_mask.shape[-1]).to(q_bh.dtype)
                scale = getattr(attn, "scale", 1.0 / (head_dim ** 0.5))
                out_bh = unet_controller.xattn_apply_kv_cache(
                    q_bh, k_bh, v_bh, scale,
                    time_step=ts,
                    unet_position=pos,
                    attn_index=attn_index,
                    # source_tag="text",
                    source_tag="text_mask_ip",
                    do_uncond=True,
                    batch_index=None,
                    attention_mask=mask_bhq,
                    hist_topk=64,  # top-k most related 64 keys
                    hist_score_bias=0.35,
                )

                hidden_states = out_bh.view(batch_size, attn.heads, -1, head_dim)
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
            else:
                hidden_states = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
        else:
            new_query = attn.head_to_batch_dim(query)
            key_bh = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            value_bh = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
            _mask = None
            if attention_mask is not None:
                _mask = attention_mask.view(batch_size * attn.heads, -1, attention_mask.shape[-1])
            attention_probs = attn.get_attention_scores(new_query, key_bh, _mask)


            hidden_states = torch.bmm(attention_probs, value_bh)

            hidden_states = attn.batch_to_head_dim(hidden_states)


        # get psuedo attention mask for image: better start after some timesteps
        psuedo_attention_mask, psuedo_dummy_attention_mask = self.get_psuedo_attention_mask(attn.heads)
        psuedo_attention_mask, post_psuedo_dummy_attention_mask = self.get_psuedo_attention_mask(attn.heads)
        # ...  psuedo_attention_mask reshape  [B,H,Q,K]

        if psuedo_attention_mask is not None:
            psuedo_attention_mask = psuedo_attention_mask.view(batch_size, attn.heads, -1,
                                                               psuedo_attention_mask.shape[-1])



        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        rf_attention_mask = attention_mask_qk_image if attention_mask_qk_image is not None else rf_attention_mask
        rf_attention_mask = psuedo_attention_mask if psuedo_attention_mask is not None else rf_attention_mask
        dummy_attention_mask = psuedo_dummy_attention_mask if psuedo_dummy_attention_mask is not None else dummy_attention_mask

        # === NEW: map-derived bias & sigma scales ===
        map_bias, sigma_scales = self._build_map_bias_and_sigma_scales(
            head_size=attn.heads,
            num_dummy_tokens=self.num_dummy_tokens,
            num_tokens_per_subject=self.num_tokens,
            seq_len_q=hidden_states.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if map_bias is not None:
            rf_attention_mask = (rf_attention_mask if rf_attention_mask is not None else 0.0) + map_bias
        self._sigma_scales = sigma_scales


        if not self.need_image_attention_map:
            new_query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            ip_hidden_states = F.scaled_dot_product_attention(
                new_query, ip_key, ip_value, attn_mask=rf_attention_mask, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
        else:
            new_query = attn.head_to_batch_dim(query)
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            if rf_attention_mask is not None:
                rf_attention_mask = rf_attention_mask.view(batch_size * attn.heads, -1, rf_attention_mask.shape[-1])


            ip_attention_probs = attn.get_attention_scores(new_query, ip_key, rf_attention_mask)

            ip_attention_probs = torch.where(dummy_attention_mask.unsqueeze(-1), 0.0, ip_attention_probs)

            # ===== new IP attention refresh center =====
            if boxes is not None:
                try:
                    attn_centers = self._compute_ip_centers(
                        ip_attention_probs=ip_attention_probs,
                        boxes=boxes,
                        batch_size=batch_size,
                        head_size=attn.heads,
                        seq_len_q=hidden_states.shape[1],
                        num_dummy_tokens=self.num_dummy_tokens,
                        num_tokens_per_subject=self.num_tokens,
                        ema=0.6,
                    )

                    custom_attention_masks = self.prepare_attention_mask_qk_two_stage_gaussian_new(
                        boxes,
                        phrase_idxes,
                        hidden_states.shape[1],
                        self.text_tokens,
                        batch_size,
                        attn.heads,
                        hidden_states.dtype,
                        hidden_states.device,
                        use_masked_text_attention=False,
                        attn_centers_override=attn_centers,
                    )
                    attention_mask_qk_image, _, dummy_attention_mask = custom_attention_masks

                    if attention_mask_qk_image is not None:
                        rf_attention_mask = attention_mask_qk_image

                    if dummy_attention_mask is not None:
                        dummy_attention_mask = dummy_attention_mask.to(torch.bool)

                except Exception:
                    pass



            if self.subject_scales is not None:
                # apply different scales to different subjects
                subject_scales = torch.tensor(self.subject_scales, dtype=ip_attention_probs.dtype,
                                              device=ip_attention_probs.device)
                subject_scales = subject_scales.unsqueeze(0).unsqueeze(0).repeat_interleave(self.num_tokens, dim=-1)
                dummy_subject_scales = torch.ones((1, 1, 1), dtype=ip_attention_probs.dtype,
                                                  device=ip_attention_probs.device).repeat_interleave(
                    self.num_dummy_tokens, dim=-1)
                subject_scales = torch.cat([dummy_subject_scales, subject_scales], dim=-1)
                ip_attention_probs = ip_attention_probs * subject_scales


            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)


        if self.subject_scales is None:
            hidden_states = hidden_states + self.scale * ip_hidden_states
        else:
            hidden_states = hidden_states + ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)



        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CNAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, num_tokens=4, text_tokens=77):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_tokens = num_tokens
        self.text_tokens = text_tokens

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        rf_attention_mask = None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            end_pos = self.text_tokens
            encoder_hidden_states = encoder_hidden_states[:, :end_pos]  # only use text
            attention_mask = attention_mask[:, :, :end_pos]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states