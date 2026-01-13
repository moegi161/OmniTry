import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

#try:
#    from flash_attn import flash_attn_varlen_func
#    FLASH_ATTN_AVALIABLE = True
#except:
FLASH_ATTN_AVALIABLE = False


def build_shared_kv(
    key: torch.Tensor,          # (B, H, L, D)
    value: torch.Tensor,        # (B, H, L, D)
    txt_len: int,
    donors: list[list[int]],    # donors[i] = list of batch indices whose IMAGE tokens are appended to i
):
    """
    Returns:
      key_new:   (B, H, L_max, D)
      value_new: (B, H, L_max, D)
      k_lens:    (B,) lengths before padding
    """
    B, H, L, D = key.shape
    device = key.device

    assert value.shape == key.shape
    assert len(donors) == B, "donors must be a list of length B"
    assert 0 <= txt_len <= L, f"txt_len={txt_len} must be within [0, L={L}]"

    key_list = []
    val_list = []

    for i in range(B):
        # Start with sample i’s full tokens (text + image): (H, L, D)
        k_i = key[i]
        v_i = value[i]

        # Append donors’ image tokens only: (H, L - txt_len, D)
        if donors[i]:
            k_app = [key[j, :, txt_len:] for j in donors[i]]
            v_app = [value[j, :, txt_len:] for j in donors[i]]
            k_i = torch.cat([k_i] + k_app, dim=1)
            v_i = torch.cat([v_i] + v_app, dim=1)

        # (H, seq, D) -> (seq, H, D) for pad_sequence
        key_list.append(k_i.permute(1, 0, 2))
        val_list.append(v_i.permute(1, 0, 2))

    k_lens = torch.tensor([t.shape[0] for t in key_list], device=device, dtype=torch.long)

    # (B, L_max, H, D) -> (B, H, L_max, D)
    key_new = pad_sequence(key_list, batch_first=True).permute(0, 2, 1, 3)
    val_new = pad_sequence(val_list, batch_first=True).permute(0, 2, 1, 3)

    return key_new, val_new, k_lens



def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis,
    use_real = True,
    use_real_unbind_dim = -1,
):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([B, S, D], [B, S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        B, H, S, D = x.size()
        cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
        cos = cos.unsqueeze(1)  
        sin = sin.unsqueeze(1)
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        lens=None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        #print("Shapes after linear projection:", query.shape, key.shape, value.shape)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        #print("Shapes after projection:", query.shape, key.shape, value.shape)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
            
            #print("Shapes after adding encoder projections:", query.shape, key.shape, value.shape)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # supporting sequence length
        q_lens = lens.clone() if lens is not None else torch.LongTensor([query.shape[2]] * batch_size).to(query.device)
        k_lens = lens.clone() if lens is not None else torch.LongTensor([key.shape[2]] * batch_size).to(key.device)


        #print("before hacked shared attention:", query.shape, key.shape, value.shape, q_lens, k_lens)
        """
        # hacked: shared attention
        txt_len = 512
        context_key = [
            torch.cat([key[0], key[1, :, txt_len:]], dim=1).permute(1, 0, 2),
            key[1].permute(1, 0, 2)
        ]
        context_value = [
            torch.cat([value[0], value[1, :, txt_len:]], dim=1).permute(1, 0, 2),
            value[1].permute(1, 0, 2)
        ]
        k_lens = torch.LongTensor([k.size(0) for k in context_key]).to(query.device)
        key = pad_sequence(context_key, batch_first=True).permute(0, 2, 1, 3)
        value = pad_sequence(context_value, batch_first=True).permute(0, 2, 1, 3)
        """
        
        # shared attention / temporal attention across batch:
        # assume batch = (target_0, ..., target_{T-1}, reference)
        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1] #Ianna: length of text tokens, originally hardcoded to 512
        else:
            txt_len = 0

        
        B = key.shape[0]
        ref_idx = B - 1
        T = B - 1  # number of targets

        # Example: each target attends to all other targets + reference
        donors = [[] for _ in range(B)]
        for i in range(T):
            donors[i] = [ref_idx]   # target i sees reference only
            # all other targets + ref
            #neighbors = [j for j in range(T) if j != i]
            #donors[i] = neighbors + [ref_idx]
        
        # keep reference unchanged (or set donors[ref_idx] if you want symmetric coupling)
        donors[ref_idx] = []

        """    
        B, H, L, D = key.shape
        assert B % 2 == 0, "Expect (target, ref, target, ref, ...)"
        num_pairs = B // 2
        t_indices = [2 * p for p in range(num_pairs)]
        r_indices = [2 * p + 1 for p in range(num_pairs)]

        donors = [[] for _ in range(B)]
        for p in range(num_pairs):
            t_idx = 2 * p
            r_idx = 2 * p + 1

            other_targets = [t for t in t_indices if t != t_idx] 

            donors[t_idx] = [r_idx] #+ other_targets   # tar gets: its ref 
            donors[r_idx] = []                        # refs stay passive

        """
        key, value, k_lens = build_shared_kv(key, value, txt_len=txt_len, donors=donors)
        
        
        # core attention
        if FLASH_ATTN_AVALIABLE:
            query = query.permute(0, 2, 1, 3)   # batch, sequence, num_head, head_dim
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            
            query = torch.cat([u[:l] for u, l in zip(query, q_lens)], dim=0)
            key = torch.cat([u[:l] for u, l in zip(key, k_lens)], dim=0)
            value = torch.cat([u[:l] for u, l in zip(value, k_lens)], dim=0)
            cu_seqlens_q = F.pad(q_lens.cumsum(dim=0), (1, 0)).to(torch.int32)
            cu_seqlens_k = F.pad(k_lens.cumsum(dim=0), (1, 0)).to(torch.int32)
            max_seqlen_q = torch.max(q_lens).item()
            max_seqlen_k = torch.max(k_lens).item()
            
            #print("Shapes before flash attention:", query.shape, key.shape, value.shape)

            hidden_states = flash_attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
            hidden_states = pad_sequence([
                hidden_states[start: end]
                for start, end in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])
            ], batch_first=True)
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            attn_mask = torch.zeros((query.size(0), 1, query.size(2), key.size(2)), dtype=torch.bool).to(query)
            for i, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
                attn_mask[i, :, :q_len, :k_len] = True
                
            #print("Shapes before scaled dot product attention:", query.shape, key.shape, value.shape, attn_mask.shape)

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)
        
        # ===================== Ianna: Stage-B temporal attention (targets <-> targets, no params) =====================
        donors_B = [[] for _ in range(B)]
        for i in range(T):
            # all other targets
            neighbors = [j for j in range(T) if abs(j - i) <= 1 and j != i]  # window size = 2
            donors_B[i] = neighbors
        # refs stay passive: donors_B[ref_idx] = [] by construction

        # Use Stage-A output as input to temporal attention
        hidden_states_stage1 = hidden_states  # (B, L_max, H*D)

         # ===================== Ianna: Stage-B temporal attention (targets <-> targets, memory-efficient) =====================
        # donors_B is already built above:
        #   donors_B[i] = all other targets for target i
        #   donors_B[ref_idx] remains []  (refs passive)

        hidden_states_stage1 = hidden_states  # (B, L_max, H*D)

        B2, L_max, HD = hidden_states_stage1.shape
        H = attn.heads
        head_dim = HD // H
        assert B2 == B, "Batch size mismatch between Stage A and Stage B"

        # Treat Stage-A hidden states as Q=K=V (no extra projections, no learnable params)
        # (B, L_max, H*D) -> (B, H, L_max, D)
        qkv = hidden_states_stage1.view(B2, L_max, H, head_dim).permute(0, 2, 1, 3)
        query_t = qkv
        key_t   = qkv
        value_t = qkv

        # Build shared K/V across targets only
        key_t, value_t, k_lens_B = build_shared_kv(
            key_t, value_t, txt_len=txt_len, donors=donors_B
        )
        q_lens_B = q_lens  # same query lengths as Stage A

        # Core temporal attention (reuse FlashAttention / SDPA, no manual QK^T)
        if FLASH_ATTN_AVALIABLE:
            # flash-attn expects (B, L, H, D)
            query_fb = query_t.permute(0, 2, 1, 3)  # (B, L_max, H, D)
            key_fb   = key_t.permute(0, 2, 1, 3)    # (B, L_max_K, H, D)
            value_fb = value_t.permute(0, 2, 1, 3)

            # pack variable-length sequences
            packed_q = torch.cat([u[:l] for u, l in zip(query_fb, q_lens_B)], dim=0)
            packed_k = torch.cat([u[:l] for u, l in zip(key_fb,   k_lens_B)], dim=0)
            packed_v = torch.cat([u[:l] for u, l in zip(value_fb, k_lens_B)], dim=0)

            cu_seqlens_q_B = F.pad(q_lens_B.cumsum(dim=0), (1, 0)).to(torch.int32)
            cu_seqlens_k_B = F.pad(k_lens_B.cumsum(dim=0), (1, 0)).to(torch.int32)
            max_seqlen_q_B = int(q_lens_B.max().item())
            max_seqlen_k_B = int(k_lens_B.max().item())

            hs_flat = flash_attn_varlen_func(
                packed_q,
                packed_k,
                packed_v,
                cu_seqlens_q_B,
                cu_seqlens_k_B,
                max_seqlen_q_B,
                max_seqlen_k_B,
            )  # (sum_q, H, D)

            # unpack back to (B, L_max, H, D)
            hs_padded = pad_sequence(
                [
                    hs_flat[start:end]
                    for start, end in zip(cu_seqlens_q_B[:-1], cu_seqlens_q_B[1:])
                ],
                batch_first=True,
            )  # (B, L_max, H, D)
            hidden_states_B = hs_padded.reshape(B2, L_max, H * head_dim)

        else:
            # SDPA path: query_t/key_t/value_t are (B, H, L, D)
            attn_mask_B = torch.zeros(
                (B2, 1, query_t.size(2), key_t.size(2)),
                dtype=torch.bool,
                device=query_t.device,
            )
            for i, (q_len_i, k_len_i) in enumerate(zip(q_lens_B, k_lens_B)):
                attn_mask_B[i, :, :q_len_i, :k_len_i] = True

            hs = F.scaled_dot_product_attention(
                query_t, key_t, value_t,
                attn_mask=attn_mask_B,
                dropout_p=0.0,
                is_causal=False,
            )  # (B, H, L_max, D)

            hidden_states_B = hs.transpose(1, 2).reshape(
                B2, L_max, H * head_dim
            )  # (B, L_max, H*D)

        hidden_states_B = hidden_states_B.to(hidden_states_stage1.dtype)

        # Make reference truly passive: keep its Stage-A representation
        ref_len = q_lens_B[ref_idx].item()
        hidden_states_B[ref_idx, :ref_len] = hidden_states_stage1[ref_idx, :ref_len]

        # Blend Stage-B with Stage-A via a residual to avoid over-smoothing
        lambda_t = 0.7  # you can tune this (0.1–0.7)
        hidden_states = hidden_states_stage1 + lambda_t * (hidden_states_B - hidden_states_stage1)
        # ===================== End Stage-B temporal attention =====================



        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
