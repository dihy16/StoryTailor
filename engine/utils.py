#modified from 1p1s
import contextlib

import torch
from typing import Optional
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler, EDMDPMSolverMultistepScheduler
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from scipy.spatial.distance import cdist
import numpy as np
from . import pipeline_stable_diffusion_xl
from torch.fft import fftn, fftshift, ifftn, ifftshift
from typing import Optional, Tuple

from .unet_controller import UNetController


def _token_component_energy(s, vh):
    # s: [k], vh: [k, n_tokens]
    # energy[j, i] = (s[i] * vh[i, j])^2  -> return [n_tokens, k]
    return (vh.transpose(0, 1) * s.unsqueeze(0)).pow(2)

def find_sublist(full_list, sub_list):
    """
    Find the start index of sub_list inside full_list.
    """
    n = len(sub_list)
    for i in range(len(full_list) - n + 1):
        if full_list[i:i+n] == sub_list:
            return i
    return -1


def gen_layout_attention_mask(query, key, boxes, phrase_idxes, eot_idxes, device, dtype):
    """
    生成 layout-guided cross-attention mask。
    Args:
        query: (batch, heads, query_len, head_dim)
        key: (batch, heads, key_len, head_dim)
        boxes: List[List[List[4 floats]]] -> [batch, subjects, box coordinates]
        phrase_idxes: List[List[int]] -> [batch, subjects]
        eot_idxes: List[List[int]] -> [batch, subjects]
        device: device
        dtype: dtype
    Returns:
        attention_mask: Tensor (batch, heads, query_len, key_len)
    """
    batch_size, num_heads, query_len, _ = query.shape
    _, _, key_len, _ = key.shape

    attention_mask = torch.zeros(
        (batch_size, num_heads, query_len, key_len),
        device=device,
        dtype=dtype,
    )

    for b in range(batch_size):
        subject_boxes = boxes[b]  # [[x1, y1, x2, y2], ...]
        subject_phrases = phrase_idxes[b]  # [idx1, idx2, ...]
        subject_eots = eot_idxes[b]  # [eot1, eot2, ...]

        for subj_idx, (box, phrase_idx, eot_idx) in enumerate(zip(subject_boxes, subject_phrases, subject_eots)):
            # 允许query attend到自己的box区域 + 对应的text token (phrase_idx到eot_idx之间)
            # 在这里简单处理：允许 attend 到 phrase_idx位置
            if 0 <= phrase_idx < key_len:
                attention_mask[b, :, :, phrase_idx] = 0  # allow attending
            # 有些情况需要扩展，可以让query attend到整个短语或多个tokens（比如 phrase_idx ~ eot_idx）

    # 默认是零mask（允许所有），为了mask掉无关的，我们需要设置-10000到非allowed的地方
    # (Softmax(x + mask)，mask中很小的值，相当于屏蔽掉)
    attention_mask = (attention_mask - 1) * 10000.0

    return attention_mask

def ipca(q, k, v, scale, unet_controller: Optional[UNetController] = None): # eg. q: [4,20,1024,64] k,v: [4,20,77,64] 
    q_neg, q_pos = torch.split(q, q.size(0) // 2, dim=0)
    k_neg, k_pos = torch.split(k, k.size(0) // 2, dim=0)
    v_neg, v_pos = torch.split(v, v.size(0) // 2, dim=0)

    # 1. negative_attn

    scores_neg = torch.matmul(q_neg, k_neg.transpose(-2, -1)) * scale
    attn_weights_neg = torch.softmax(scores_neg, dim=-1)
    attn_output_neg = torch.matmul(attn_weights_neg, v_neg)

    # 2. positive_attn (we do ipca only on positive branch)

    print(k_pos.shape)
    # 2.1 ipca 
    k_plus = torch.cat(tuple(k_pos.transpose(-2, -1)), dim=2).unsqueeze(0).repeat(k_pos.size(0),1,1,1) # 𝐾+ = [𝐾1 ⊕ 𝐾2 ⊕ . . . ⊕ 𝐾𝑁 ]
    v_plus = torch.cat(tuple(v_pos), dim=1).unsqueeze(0).repeat(v_pos.size(0),1,1,1) # 𝑉+ = [𝑉1 ⊕ 𝑉2 ⊕ . . . ⊕ 𝑉𝑁 ]


    # 2.2 apply mask
    if unet_controller is not None:
        scores_pos = torch.matmul(q_pos, k_plus) * scale

        
        # 2.2.1 apply dropout mask
        dropout_mask = gen_dropout_mask(scores_pos.shape, unet_controller, unet_controller.Ipca_dropout) # eg: [a,1024,154]   


        # 2.2.3 apply embeds mask
        if unet_controller.Use_embeds_mask:
            apply_embeds_mask(unet_controller,dropout_mask, add_eot=False)

        mask = dropout_mask

        mask = mask.unsqueeze(1).repeat(1,scores_pos.size(1),1,1)
        attn_weights_pos = torch.softmax(scores_pos + torch.log(mask), dim=-1)

    else:
        scores_pos = torch.matmul(q_pos, k_plus) * scale
        attn_weights_pos = torch.softmax(scores_pos, dim=-1)


    attn_output_pos = torch.matmul(attn_weights_pos, v_plus)
    # 3. combine
    attn_output = torch.cat((attn_output_neg, attn_output_pos), dim=0)

    return attn_output

def ipca2(q, k, v, scale, unet_controller: Optional[UNetController] = None): # eg. q: [4,20,1024,64] k,v: [4,20,77,64] 
    if unet_controller.ipca_time_step != unet_controller.current_time_step:
        unet_controller.ipca_time_step = unet_controller.current_time_step
        unet_controller.ipca2_index = 0
    else:
        unet_controller.ipca2_index += 1

    if unet_controller.Store_qkv is True:

        key = f"cross {unet_controller.current_time_step} {unet_controller.current_unet_position} {unet_controller.ipca2_index}"
        unet_controller.k_store[key] = k
        unet_controller.v_store[key] = v

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
    else:
        # batch > 1
        if unet_controller.frame_prompt_express_list is not None:
            batch_size = q.size(0) // 2
            attn_output_list = []

            for i in range(batch_size):
                q_i = q[[i, i + batch_size], :, :, :]
                k_i = k[[i, i + batch_size], :, :, :]
                v_i = v[[i, i + batch_size], :, :, :]

                q_neg_i, q_pos_i = torch.split(q_i, q_i.size(0) // 2, dim=0)
                k_neg_i, k_pos_i = torch.split(k_i, k_i.size(0) // 2, dim=0)
                v_neg_i, v_pos_i = torch.split(v_i, v_i.size(0) // 2, dim=0)

                key = f"cross {unet_controller.current_time_step} {unet_controller.current_unet_position} {unet_controller.ipca2_index}"
                q_store = q_i
                k_store = unet_controller.k_store[key]
                v_store = unet_controller.v_store[key]

                q_store_neg, q_store_pos = torch.split(q_store, q_store.size(0) // 2, dim=0)
                k_store_neg, k_store_pos = torch.split(k_store, k_store.size(0) // 2, dim=0)
                v_store_neg, v_store_pos = torch.split(v_store, v_store.size(0) // 2, dim=0)    

                q_neg = torch.cat((q_neg_i, q_store_neg), dim=0)
                q_pos = torch.cat((q_pos_i, q_store_pos), dim=0)
                k_neg = torch.cat((k_neg_i, k_store_neg), dim=0)
                k_pos = torch.cat((k_pos_i, k_store_pos), dim=0)
                v_neg = torch.cat((v_neg_i, v_store_neg), dim=0)
                v_pos = torch.cat((v_pos_i, v_store_pos), dim=0)

                q_i = torch.cat((q_neg, q_pos), dim=0)
                k_i = torch.cat((k_neg, k_pos), dim=0)
                v_i = torch.cat((v_neg, v_pos), dim=0)

                attn_output_i = ipca(q_i, k_i, v_i, scale, unet_controller)
                attn_output_i = attn_output_i[[0, 2], :, :, :]
                attn_output_list.append(attn_output_i)
            
            attn_output_ = torch.cat(attn_output_list, dim=0)
            attn_output = torch.zeros(size=(q.size(0), attn_output_i.size(1), attn_output_i.size(2), attn_output_i.size(3)), device=q.device, dtype=q.dtype)
            for i in range(batch_size):
                attn_output[i] = attn_output_[i*2]
            for i in range(batch_size):
                attn_output[i + batch_size] = attn_output_[i*2 + 1]
        # batch = 1
        else:
            q_neg, q_pos = torch.split(q, q.size(0) // 2, dim=0)
            k_neg, k_pos = torch.split(k, k.size(0) // 2, dim=0)
            v_neg, v_pos = torch.split(v, v.size(0) // 2, dim=0)

            key = f"cross {unet_controller.current_time_step} {unet_controller.current_unet_position} {unet_controller.ipca2_index}"
            q_store = q
            k_store = unet_controller.k_store[key]
            v_store = unet_controller.v_store[key]

            q_store_neg, q_store_pos = torch.split(q_store, q_store.size(0) // 2, dim=0)
            k_store_neg, k_store_pos = torch.split(k_store, k_store.size(0) // 2, dim=0)
            v_store_neg, v_store_pos = torch.split(v_store, v_store.size(0) // 2, dim=0)    

            q_neg = torch.cat((q_neg, q_store_neg), dim=0)
            q_pos = torch.cat((q_pos, q_store_pos), dim=0)
            k_neg = torch.cat((k_neg, k_store_neg), dim=0)
            k_pos = torch.cat((k_pos, k_store_pos), dim=0)
            v_neg = torch.cat((v_neg, v_store_neg), dim=0)
            v_pos = torch.cat((v_pos, v_store_pos), dim=0)

            q = torch.cat((q_neg, q_pos), dim=0)
            k = torch.cat((k_neg, k_pos), dim=0)
            v = torch.cat((v_neg, v_pos), dim=0)

            attn_output = ipca(q, k, v, scale, unet_controller)
            attn_output = attn_output[[0, 2], :, :, :]
    
    return attn_output


def apply_embeds_mask(unet_controller: Optional[UNetController],dropout_mask, add_eot=False):   
    id_prompt = unet_controller.id_prompt
    prompt_tokens = prompt2tokens(unet_controller.tokenizer,unet_controller.prompts[0])
    
    words_tokens = prompt2tokens(unet_controller.tokenizer,id_prompt)
    words_tokens = [word for word in words_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
    index_of_words = find_sublist_index(prompt_tokens,words_tokens)    
    index_list = [index+77 for index in range(index_of_words, index_of_words+len(words_tokens))]
    if add_eot:
        index_list.extend([index+77 for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>'])

    mask_indices = torch.arange(dropout_mask.size(-1), device=dropout_mask.device)
    mask = (mask_indices >= 78) & (~torch.isin(mask_indices, torch.tensor(index_list, device=dropout_mask.device)))
    dropout_mask[0, :, mask] = 0


def gen_dropout_mask(out_shape, unet_controller: Optional[UNetController], drop_out):
    gen_length = out_shape[3]
    attn_map_side_length = out_shape[2]

    batch_num = out_shape[0]
    mask_list = []
    
    for prompt_index in range(batch_num):
        start = prompt_index * int(gen_length / batch_num)
        end = (prompt_index + 1) * int(gen_length / batch_num)
    
        mask = torch.bernoulli(torch.full((attn_map_side_length,gen_length), 1 - drop_out, dtype=unet_controller.torch_dtype, device=unet_controller.device))        
        mask[:, start:end] = 1

        mask_list.append(mask)

    concatenated_mask = torch.stack(mask_list, dim=0)
    return concatenated_mask





def get_max_window_length(unet_controller: Optional[UNetController],id_prompt, frame_prompt_list):
    single_long_prompt = id_prompt
    max_window_length = 0
    for index, movement in enumerate(frame_prompt_list):
        single_long_prompt += ' ' + movement
        token_length = len(single_long_prompt.split())
        if token_length >= 77:
            break
        max_window_length += 1
    return max_window_length


def movement_gen_story_slide_windows(id_prompt,frame_prompt_list, pipe, window_length, seed, unet_controller: Optional[UNetController], save_dir, verbose=True):
    import os
    max_window_length = get_max_window_length(unet_controller,id_prompt,frame_prompt_list)
    window_length = min(window_length,max_window_length)
    if window_length < len(frame_prompt_list):
        movement_lists = circular_sliding_windows(frame_prompt_list, window_length)
    else:
        movement_lists = [movement for movement in frame_prompt_list]
    story_images = []


    if verbose: 
        print("seed:", seed)
    generate = torch.Generator().manual_seed(seed)
    unet_controller.id_prompt = id_prompt

    for index, movement in enumerate(frame_prompt_list):
        if unet_controller is not None:
            if window_length < len(frame_prompt_list):
                unet_controller.frame_prompt_suppress = movement_lists[index][1:]
                unet_controller.frame_prompt_express = movement_lists[index][0]
                gen_propmts = [f'{id_prompt} {" ".join(movement_lists[index])}']

            else:
                unet_controller.frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
                unet_controller.frame_prompt_express = movement_lists[index]
                gen_propmts = [f'{id_prompt} {" ".join(movement_lists)}']
            
            if verbose:
                print(f"suppress: {unet_controller.frame_prompt_suppress}")
                print(f"express: {unet_controller.frame_prompt_express}")
                print(f'id_prompt: {id_prompt}')
                print(f"gen_propmts: {gen_propmts}")


        else:
            gen_propmts = f'{id_prompt} {movement}'

        if unet_controller is not None and unet_controller.Use_same_init_noise is True:     
            generate = torch.Generator().manual_seed(seed)

        # images = pipe(gen_propmts, generator=generate, unet_controller=unet_controller).images
        images = pipe(
            gen_propmts,
            generator=generate,
            unet_controller=unet_controller,
            # below: new args
            reset_prev_latents=(index == 0),
            use_last_x0_as_init=(index > 0),
            view_jitter=0.03,  # 先给个温和默认，后面你可以自由调
            view_jitter_mode="affine",
            cache_prev_on_cpu=True,  # 显存紧张就 True；想更快可 False
        ).images

        story_images.append(images[0])
        images[0].save(os.path.join(save_dir, f'{id_prompt} {unet_controller.frame_prompt_express}.jpg'))


    image_array_list = [np.array(pil_img) for pil_img in story_images]

    # Concatenate images horizontally
    story_image = np.concatenate(image_array_list, axis=1)
    story_image = Image.fromarray(story_image.astype(np.uint8))

    if unet_controller.Save_story_image:
        story_image.save(os.path.join(save_dir, f'story_image_{id_prompt}.jpg'))

    return story_images, story_image

# this function set batch > 1 to generate multiple images at once
def movement_gen_story_slide_windows_batch(id_prompt, frame_prompt_list, pipe, window_length, seed, unet_controller: Optional[UNetController], save_dir, batch_size=3):  
    import os
    max_window_length = get_max_window_length(unet_controller,id_prompt,frame_prompt_list)
    window_length = min(window_length,max_window_length)
    if window_length < len(frame_prompt_list):
        movement_lists = circular_sliding_windows(frame_prompt_list, window_length)
    else:
        movement_lists = [movement for movement in frame_prompt_list]
    story_images = []

    print("seed:", seed)
    generate = torch.Generator().manual_seed(seed)
    unet_controller.id_prompt = id_prompt

    gen_prompt_info_list = []
    gen_prompt = None
    for index, _ in enumerate(frame_prompt_list):
        if window_length < len(frame_prompt_list):
            frame_prompt_suppress = movement_lists[index][1:]
            frame_prompt_express = movement_lists[index][0]
            gen_prompt = f'{id_prompt} {" ".join(movement_lists[index])}'

        else:
            frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
            frame_prompt_express = movement_lists[index]
            gen_prompt = f'{id_prompt} {" ".join(movement_lists)}'

        gen_prompt_info_list.append({'frame_prompt_suppress': frame_prompt_suppress, 'frame_prompt_express': frame_prompt_express})
    
    story_images = []
    for i in range(0, len(gen_prompt_info_list), batch_size):
        batch = gen_prompt_info_list[i:i + batch_size]
        gen_prompts = [gen_prompt for _ in batch]
        unet_controller.frame_prompt_express_list = [gen_prompt_info['frame_prompt_express'] for gen_prompt_info in batch]
        unet_controller.frame_prompt_suppress_list = [gen_prompt_info['frame_prompt_suppress'] for gen_prompt_info in batch]
                
        if unet_controller is not None and unet_controller.Use_same_init_noise is True:     
            generate = torch.Generator().manual_seed(seed)
        
        images = pipe(gen_prompts, generator=generate, unet_controller=unet_controller).images
        for index,image in enumerate(images):
            story_images.append(image)
            image.save(os.path.join(save_dir, f'{id_prompt} {unet_controller.frame_prompt_express_list[index]}.jpg'))

    image_array_list = [np.array(pil_img) for pil_img in story_images]

    # Concatenate images horizontally
    story_image = np.concatenate(image_array_list, axis=1)
    story_image = Image.fromarray(story_image.astype(np.uint8))

    if unet_controller.Save_story_image:
        story_image.save(os.path.join(save_dir, 'story_image.jpg'))

    return story_images, story_image


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens





@torch.no_grad()
def punish_wight_rank_Trunc(
    tensor: torch.Tensor,
    latent_size: int,
    alpha: float = 1.0,
    beta: float = 1.2,
    calc_similarity: bool = False,
    rank: int | None = None,
    token_texts: list[str] | None = None,
    log_tag: str = "absvr",
    action_token_indices: list[int] | None = None,
):

    # --------- shape / bounds ----------
    if tensor.dim() != 2:
        raise ValueError(f"expect 2D tensor [embed_dim, num_tokens], got {tuple(tensor.shape)}")
    d, n = tensor.shape
    kmax = min(d, n)
    latent_size = max(1, min(int(latent_size), kmax))
    if rank is not None:
        rank = max(1, min(int(rank), latent_size))

    # --------- SVD should be in fp32, disable autocast ----------
    X_in = tensor
    X = tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor
    ac = torch.autocast("cuda", enabled=False) if X.is_cuda else contextlib.nullcontext()

    with ac:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # U:[d,k] S:[k] Vh:[k,n]
        U = U[:, :latent_size]
        S = S[:latent_size]
        Vh = Vh[:latent_size, :]


        # --------- low-rank truncation ----------
        if rank is not None:
            S_trunc = S.clone()
            S_trunc[rank:] = 0
        else:
            S_trunc = S

        # --------- SVR-like reweight (your original formula) ----------
        S_new = S_trunc * torch.exp(-alpha * S_trunc) * beta

        # --------- reconstruct ----------
        # (U * S_new) @ Vh is faster than U @ diag(S_new) @ Vh
        X_new = (U * S_new.unsqueeze(0)) @ Vh  # [d, n]

        # ---- optional similarity check (token 0) ----
        if calc_similarity:
            dist = cdist(
                X[:, 0].unsqueeze(0).cpu().numpy(),
                X_new[:, 0].unsqueeze(0).cpu().numpy(),
                metric="cosine",
            )
            print(f"[{log_tag}] cosine distance(token0) before/after: {dist.item():.6f}")


    # cast back
    if X_in.dtype in (torch.float16, torch.bfloat16):
        X_new = X_new.to(X_in.dtype)
    return X_new

def swr_single_prompt_embeds(swr_words,prompt_embeds,prompt,tokenizer,alpha=1.0, beta=1.2, zero_eot=False, tau=0.85):
    punish_indices = []

    prompt_tokens = prompt2tokens(tokenizer,prompt)
    
    words_tokens = prompt2tokens(tokenizer,swr_words)
    words_tokens = [word for word in words_tokens if word != '<|endoftext|>' and word != '<|startoftext|>']
    index_of_words = find_sublist_index(prompt_tokens,words_tokens)
    
    if index_of_words != -1:
        punish_indices.extend([num for num in range(index_of_words, index_of_words+len(words_tokens))])
    
    if zero_eot:
        eot_indices = [index for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>']
        prompt_embeds[eot_indices] *= 9e-1
        pass
    else:
        punish_indices.extend([index for index, word in enumerate(prompt_tokens) if word == '<|endoftext|>'])

    punish_indices = list(set(punish_indices))
    
    wo_batch = prompt_embeds[punish_indices]

    wo_batch = apply_trunc_svr_with_energy(
        wo_batch,  # [n, d]
        tau=tau,
        r_min=0, r_max=32,
        alpha=alpha, beta=beta,

    )


    prompt_embeds[punish_indices] = wo_batch


def find_sublist_index(list1, list2):
    for i in range(len(list1) - len(list2) + 1):
        if list1[i:i + len(list2)] == list2:
            return i
    return -1  # If sublist is not found


def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    """Fourier filter as introduced in FreeU (https://arxiv.org/abs/2309.11497).

    This version of the method comes from here:
    https://github.com/huggingface/diffusers/pull/5164#issuecomment-1732638706
    """
    x = x_in
    B, C, H, W = x.shape

    x = x.to(dtype=torch.float32)

    # FFT
    x_freq = fftn(x, dim=(-2, -1))
    x_freq = fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype=x_in.dtype)


def apply_freeu(
    resolution_idx: int, hidden_states: "torch.Tensor", res_hidden_states: "torch.Tensor", **freeu_kwargs
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Applies the FreeU mechanism as introduced in https:
    //arxiv.org/abs/2309.11497. Adapted from the official code repository: https://github.com/ChenyangSi/FreeU.

    Args:
        resolution_idx (`int`): Integer denoting the UNet block where FreeU is being applied.
        hidden_states (`torch.Tensor`): Inputs to the underlying block.
        res_hidden_states (`torch.Tensor`): Features from the skip block corresponding to the underlying block.
        s1 (`float`): Scaling factor for stage 1 to attenuate the contributions of the skip features.
        s2 (`float`): Scaling factor for stage 2 to attenuate the contributions of the skip features.
        b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
        b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
    """
    if resolution_idx == 0:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b1"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s1"])
    if resolution_idx == 1:
        num_half_channels = hidden_states.shape[1] // 2
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b2"]
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s2"])

    return hidden_states, res_hidden_states


def circular_sliding_windows(lst, w):
    n = len(lst)
    windows = []
    for i in range(n):
        window = [lst[(i + j) % n] for j in range(w)]
        windows.append(window)
    return windows


def ensure_bhqk(t: torch.Tensor, b: int = None, h: int = None, heads_hint: int = None):
    if t.dim() == 4:
        # [B,H,Q,K]
        return t
    elif t.dim() == 3:
        d0, q, k = t.shape

        if (b is not None) and (d0 == b):
            return t.unsqueeze(1)  # -> [B,1,Q,K]

        if h is None:
            h = heads_hint
        if (h is not None) and (d0 % h == 0):
            b_infer = d0 // h
            return t.reshape(b_infer, h, q, k)
        if (b is not None) and (d0 % b == 0):
            h_infer = d0 // b
            return t.reshape(b, h_infer, q, k)
        raise ValueError(
            f"cannot convertt {t.shape} into [B,H,Q,K]"
        )
    else:
        raise ValueError(f"期望 3D/4D 注意力张量，收到 {t.dim()}D: {tuple(t.shape)}")

import contextlib
import torch

@torch.no_grad()
def _select_rank_by_energy(s: torch.Tensor, tau: float = 0.90,
                           r_min: int = 1, r_max: int | None = None,
                           use_squared: bool = True) -> int:

    e = s * s if use_squared else s
    tot = e.sum()
    if tot <= 0:
        return max(1, r_min)
    cume = torch.cumsum(e, dim=0)
    r = int(torch.searchsorted(cume, tau * tot).item()) + 1
    if r_max is not None:
        r = min(r, int(r_max))
    return max(r_min, r)

@torch.no_grad()
def apply_trunc_svr_with_energy(T: torch.Tensor,
                                tau: float = 0.90,
                                r_min: int = 1, r_max: int | None = None,
                                alpha: float = 0.03, beta: float = 1.12,
                                latent_size: int | None = None):

    device, dtype = T.device, T.dtype
    Xt = T.transpose(0, 1).contiguous()              # [d, n]
    d, n = Xt.shape
    kmax = min(d, n)

    if latent_size is None:
        latent_size = kmax
    latent_size = max(1, min(int(latent_size), kmax))
    if r_max is None:
        r_max = latent_size
    else:
        r_max = max(1, min(int(r_max), latent_size))


    Xcall = Xt.float() if Xt.dtype in (torch.float16, torch.bfloat16) else Xt
    ac = torch.autocast("cuda", enabled=False) if Xcall.is_cuda else contextlib.nullcontext()

    with ac:
        _, s, _ = torch.linalg.svd(Xcall, full_matrices=False)
        r = _select_rank_by_energy(s, tau=tau, r_min=r_min, r_max=r_max, use_squared=True)


        Yt = punish_wight_rank_Trunc(
            Xcall, latent_size=latent_size, alpha=alpha, beta=beta, rank=r
        )

    if Xt.dtype in (torch.float16, torch.bfloat16):
        Yt = Yt.to(Xt.dtype)
    return Yt.transpose(0, 1).to(device=device, dtype=dtype)


