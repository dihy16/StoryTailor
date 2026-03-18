# unet_controller.py
from collections import defaultdict
import torch, math


class UNetController():
    # ========= Static variables (Hyperparameters) =========
    Is_freeu_enabled = False
    Freeu_parm = {'s1': 0.6, 's2': 0.4, 'b1': 1.1, 'b2': 1.2}

    tau = 0.85

    # Ipca parameters
    Use_ipca = False
    # Ipca_position = ['down0', 'down1', 'down2', 'mid', 'up0', 'up1', 'up2']
    Ipca_position = ['mid', 'up0', 'up1', 'up2']
    Ipca_start_step = 0
    Ipca_dropout = 0.0
    Use_embeds_mask = False

    # ====== SVR parameters======
    # Alpha_weaken = 0.01   # 0.01~0.5
    # Beta_weaken  = 0.05   # 0.05~1.0
    # Alpha_enhance = -0.02 # -0.001~-0.02
    # Beta_enhance  = 1.8   # 1.0~2.0
    Alpha_weaken = 0.01   # 0.01~0.5
    Beta_weaken  = 0.05   # 0.05~1.0
    Alpha_enhance = -0.01 # -0.001~-0.02
    Beta_enhance  = 1.0   # 1.0~2.0

    # SVR settings
    Prompt_embeds_mode = 'svr'
    Remove_pool_embeds = False
    Prompt_embeds_start_step = 0

    # legacy flag
    Store_qkv = True

    # other settings
    Use_same_latents = True
    Use_same_init_noise = True
    Save_story_image = True
    # ======================================================

    def __init__(self):
        # -------- runtime variables --------
        self._variables = {}
        self.device = "cuda"
        self.torch_dtype = torch.float16

        self.prompts = None
        self.negative_prompt = None
        self.id_prompt = None
        self.frame_prompt_express = None
        self.frame_prompt_suppress = None
        self.frame_prompt_express_list = None
        self.frame_prompt_suppress_list = None

        self.tokenizer = None
        self.result_save_dir = None
        self.current_time_step = None
        self.do_classifier_free_guidance = None
        self.current_unet_position = None
        self.num_inference_steps = None

        # q/k/v store
        self.q_store = {}
        self.k_store = {}
        self.v_store = {}

        self.ipca2_index = -1
        self.ipca_time_step = -1

        # ================== accumulate（new） ==================
        self.CacheMode = "accumulate"
        self.ExcludeSources = {"ip"}
        self.CacheCapTokens = 512
        self.CacheDecay = "fifo"
        self.CacheKeyWithBatch = False
        self.CacheKeyWithDoUncond = False
        self.frame_index = -1

        # ==================  Context 3D ==================
        self.dummy_background_cache = defaultdict(list)  # key: layer_name
        self.max_cache_len = 3
        self.decay_weights = torch.tensor([1.0, 0.7, 0.5])

        # ====== Context-Cache（SDPA） ======
        self.story_auto: bool = True
        self.ctx_layers = ("down_blocks.0", "down_blocks.1", "down_blocks.2", "mid_block")
        self.ctx_alpha: float = 0.6
        self.ctx_ratio: float = 0.8
        self.ctx_place_mode: str = "interleave"

        self._mem_ctx3 = defaultdict(lambda: None)  # layer -> [1,Q,D]
        self._snap_ctx3 = {}
        self._wbuf_ctx3 = {}
        self._frame_active = False
        # -----------------------------------

    # ================== tools ==================
    def print_attributes(self):
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")

    # ================== frame cycle ==================
    @torch.no_grad()
    def begin_frame(self):
        self._snap_ctx3 = {ln: (t.clone() if t is not None else None)
                           for ln, t in self._mem_ctx3.items()}
        self._wbuf_ctx3.clear()
        self._frame_active = True
        self.frame_index += 1

    @torch.no_grad()
    def end_frame(self):
        for ln, t in self._wbuf_ctx3.items():
            self._mem_ctx3[ln] = t.detach()
        self._snap_ctx3.clear()
        self._wbuf_ctx3.clear()
        self._frame_active = False

    @torch.no_grad()
    def reset_memory(self):
        self._mem_ctx3.clear()
        self._snap_ctx3.clear()
        self._wbuf_ctx3.clear()
        self._frame_active = False
        self.frame_index = -1
        self.clear_kv_cache()

    # ================== KV cache ==================
    def set_cache_mode(self, mode: str):
        assert mode in {"off", "write_only", "accumulate"}
        self.CacheMode = mode

    def clear_kv_cache(self):
        self.q_store.clear(); self.k_store.clear(); self.v_store.clear()

    def _cap_tokens(self, x: torch.Tensor, cap: int) -> torch.Tensor:
        if cap is None or cap <= 0 or x.shape[-2] <= cap:
            return x
        if self.CacheDecay == "fifo":
            return x[:, -cap:, :]

        idx = torch.linspace(0, x.shape[-2]-1, cap, device=x.device).round().long()
        return x.index_select(dim=-2, index=idx)

    def _make_kv_key(self, time_step, unet_position: str, attn_index,
                     source_tag: str = "text", batch_index: int | None = None,
                     do_uncond: bool | None = None) -> str:
        def _part(x):
            if x is None:
                return "-1"
            if isinstance(x, int):
                return str(x)
            if torch.is_tensor(x):
                return str(int(x.detach().flatten()[0].item())) if x.numel() > 0 else "-1"
            try:
                return str(int(x))
            except Exception:
                return str(x)

        parts = [
            "xattn",
            f"t={_part(time_step)}",
            f"pos={unet_position}",
            f"i={_part(attn_index)}",
            f"src={source_tag}",
        ]
        if self.CacheKeyWithBatch and (batch_index is not None):
            parts.append(f"b={int(batch_index)}")
        if self.CacheKeyWithDoUncond and (do_uncond is not None):
            parts.append(f"u={int(bool(do_uncond))}")
        return "|".join(parts)

    # ================== kv_cache_entrance ==================
    def xattn_apply_kv_cache(self,
                             q: torch.Tensor,
                             k: torch.Tensor,
                             v: torch.Tensor,
                             scale: torch.Tensor | float,
                             *,
                             time_step: int,
                             unet_position: str,
                             attn_index: int,
                             source_tag: str = "text",
                             do_uncond: bool = False,
                             batch_index: int | None = None,
                             attention_mask: torch.Tensor | None = None,
                             hist_topk: int | None = None,
                             hist_score_bias: float | None = None) -> torch.Tensor:
        if (not self.Store_qkv) or (self.CacheMode == "off") or (source_tag in self.ExcludeSources):
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            w = torch.softmax(scores, dim=-1)
            return torch.matmul(w, v)

        if not hasattr(self, "HistTopK"):
            self.HistTopK = 0
        if not hasattr(self, "HistScoreBias"):
            self.HistScoreBias = 0.0
        hist_topk = self.HistTopK if (hist_topk is None) else hist_topk
        hist_score_bias = self.HistScoreBias if (hist_score_bias is None) else hist_score_bias

        key = self._make_kv_key(time_step, unet_position, attn_index,
                                source_tag=source_tag, batch_index=batch_index,
                                do_uncond=do_uncond)

        k_hist = self.k_store.get(key, None)
        v_hist = self.v_store.get(key, None)

        use_hist = (k_hist is not None) and (v_hist is not None) and (self.CacheMode in {"accumulate"})
        if use_hist:
            if (k_hist.dtype != k.dtype) or (k_hist.device != k.device):
                k_hist = k_hist.to(dtype=k.dtype, device=k.device)
                v_hist = v_hist.to(dtype=v.dtype, device=v.device)

            # —— Top-K ——
            if (hist_topk is not None) and (hist_topk > 0) and (k_hist.shape[-2] > hist_topk):
                # [B*H,Q,D] @ [B*H,D,L_hist] -> [B*H,Q,L_hist]
                scores_hist = torch.matmul(q, k_hist.transpose(-2, -1)) * (scale if isinstance(scale, float) else scale)
                topk_idx = scores_hist.topk(k=min(hist_topk, k_hist.shape[-2]), dim=-1).indices  # [B*H,Q,K]
                idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, k_hist.shape[-1])  # [B*H,Q,K,D]
                k_sel = torch.gather(k_hist.unsqueeze(1).expand(-1, q.shape[-2], -1, -1), -2, idx_exp)  # [B*H,Q,K,D]
                v_sel = torch.gather(v_hist.unsqueeze(1).expand(-1, q.shape[-2], -1, -1), -2, idx_exp)  # [B*H,Q,K,D]

                k_hist_eff = k_sel.reshape(q.shape[0], -1, k_hist.shape[-1])  # [B*H, Q*K, D]
                v_hist_eff = v_sel.reshape(q.shape[0], -1, v_hist.shape[-1])  # [B*H, Q*K, D]
            else:
                k_hist_eff, v_hist_eff = k_hist, v_hist

            k_cat = torch.cat([k_hist_eff, k], dim=-2)
            v_cat = torch.cat([v_hist_eff, v], dim=-2)
            hist_len = k_hist_eff.shape[-2]
        else:
            k_cat, v_cat = k, v
            hist_len = 0

        # history bias
        scores = torch.matmul(q, k_cat.transpose(-2, -1)) * (scale if isinstance(scale, float) else scale)
        if hist_len > 0 and (hist_score_bias is not None) and (hist_score_bias != 0.0):
            scores[:, :, :hist_len] = scores[:, :, :hist_len] + float(hist_score_bias)

        if attention_mask is not None:
            if attention_mask.shape[-1] != k_cat.shape[-2]:
                pad_len = k_cat.shape[-2] - attention_mask.shape[-1]
                if pad_len > 0:
                    # 用当前掩码里的最小值填充历史段（通常≈-1e4），而不是 0
                    minv = attention_mask.min()
                    pad = torch.full(
                        (attention_mask.shape[0], attention_mask.shape[1], pad_len),
                        fill_value=minv, device=attention_mask.device, dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([pad, attention_mask], dim=-1)
            scores = scores + attention_mask

        w = torch.softmax(scores, dim=-1)
        out = torch.matmul(w, v_cat)

        if self.CacheMode in {"write_only", "accumulate"} and (not do_uncond):
            k_new = self._cap_tokens(k_cat.detach(), self.CacheCapTokens)
            v_new = self._cap_tokens(v_cat.detach(), self.CacheCapTokens)
            self.k_store[key] = k_new
            self.v_store[key] = v_new

        return out

    # ================== Context tools ==================
    @staticmethod
    def _pick_ratio_mask_2d(mask_bq: torch.Tensor, ratio: float, mode: str) -> torch.Tensor:

        if ratio >= 1.0:
            return mask_bq
        B, Q = mask_bq.shape
        if Q == 0:
            return torch.zeros_like(mask_bq)
        sel = torch.zeros_like(mask_bq, dtype=torch.bool)
        if mode == "front":
            k = max(1, int(round(Q * ratio)))
            sel[:, :k] = True
        else:
            step = max(1, int(round(1.0 / max(1e-8, ratio))))
            sel[:, ::step] = True
        return mask_bq & sel

    @staticmethod
    def _ensure_bg_bq(bg_mask: torch.Tensor, B: int, Q: int, H_hint: int | None = None) -> torch.Tensor:
        if bg_mask.dim() == 2:
            if bg_mask.shape[0] == B:
                return bg_mask
            if H_hint is None or bg_mask.shape[0] != B * H_hint:
                raise ValueError(f"bg mask no matching: {tuple(bg_mask.shape)}, need H_hint={H_hint}")
            return bg_mask.view(B, H_hint, Q).any(dim=1)
        if bg_mask.dim() == 3:
            if bg_mask.shape[0] != B:
                raise ValueError(f"bg mask 1st dim B={B}，but got {bg_mask.shape[0]}")
            return bg_mask.any(dim=1)
        raise ValueError(f"unsupported bg mask shape: {tuple(bg_mask.shape)}")

    @staticmethod
    def _nearest_index_select(seq: torch.Tensor, Q_target: int) -> torch.Tensor:
        Q_src = seq.shape[1]
        if Q_src == Q_target:
            return seq
        idx = torch.linspace(0, Q_src - 1, Q_target, device=seq.device).long()
        return seq.index_select(1, idx)

    # ================== 核心：SDPA 后 3D Context 混合 ==================
    def mix_prev_context_3d(self,
                            layer_name: str,
                            ctx3: torch.Tensor,                # [B,Q,D] ← SDPA
                            bg_mask_bh_or_bhq: torch.Tensor,  # [B*H,Q] / [B,H,Q] / [B,Q]
                            H_hint: int | None = None) -> torch.Tensor:

        if not self._frame_active:
            self.begin_frame()


        if self.ctx_layers and not any(s in layer_name for s in self.ctx_layers):
            self._wbuf_ctx3[layer_name] = ctx3.detach().mean(dim=0, keepdim=True)  # [1,Q,D]
            return ctx3

        B, Q, D = ctx3.shape

        bg_bq = self._ensure_bg_bq(bg_mask_bh_or_bhq, B=B, Q=Q, H_hint=H_hint)

        bg_bq = self._pick_ratio_mask_2d(bg_bq, self.ctx_ratio, self.ctx_place_mode)  # [B,Q]

        prev = self._snap_ctx3.get(layer_name, None)  # [1,Q_prev,D] 或 None
        if prev is None:
            self._wbuf_ctx3[layer_name] = ctx3.detach().mean(dim=0, keepdim=True)
            return ctx3

        if prev.device != ctx3.device or prev.dtype != ctx3.dtype:
            prev = prev.to(device=ctx3.device, dtype=ctx3.dtype)
        prev = self._nearest_index_select(prev, Q_target=Q)     # [1,Q,D]
        if prev.shape[0] == 1 and B > 1:
            prev = prev.expand(B, -1, -1)                        # [B,Q,D]

        if bg_bq.any():
            alpha_t = torch.tensor(self.ctx_alpha, dtype=ctx3.dtype, device=ctx3.device)
            mix = bg_bq.unsqueeze(-1).to(ctx3.dtype) * alpha_t   # [B,Q,1]
            out = ctx3 * (1.0 - mix) + prev * mix
        else:
            out = ctx3

        self._wbuf_ctx3[layer_name] = ctx3.detach().mean(dim=0, keepdim=True)  # [1,Q,D]
        return out
