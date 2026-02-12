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
        self._flow_fields = None
        self._flow_donors = None
        self._ref_count = 1
        self._anchor_weight = 1.0

    def set_flow(self, flow_fields=None, flow_donors=None):
        # store flow guidance on the processor; avoids passing via kwargs that trigger warnings upstream
        self._flow_fields = flow_fields
        self._flow_donors = flow_donors

    def set_refs(self, ref_count: int = 1, anchor_weight: float = 1.0):
        self._ref_count = max(1, int(ref_count))
        self._anchor_weight = max(1.0, float(anchor_weight))

    def _run_attn_group(self, query_g, key_g, value_g, q_lens_g, k_lens_g, attn, head_dim):
        """
        query_g/key_g/value_g: (B, H_g, Lq/Lk, D)
        q_lens_g, k_lens_g: (B,)
        Returns:
            hidden_g: (B, L_max_q, H_g * D)
        """
        B = query_g.shape[0]
        H_g = query_g.shape[1]

        if FLASH_ATTN_AVALIABLE:
            # flash-attn expects (B, L, H, D)
            q_fb = query_g.permute(0, 2, 1, 3)  # (B, Lq, H_g, D)
            k_fb = key_g.permute(0, 2, 1, 3)    # (B, Lk, H_g, D)
            v_fb = value_g.permute(0, 2, 1, 3)

            # pack varlen
            q_packed = torch.cat([u[:l] for u, l in zip(q_fb, q_lens_g)], dim=0)
            k_packed = torch.cat([u[:l] for u, l in zip(k_fb, k_lens_g)], dim=0)
            v_packed = torch.cat([u[:l] for u, l in zip(v_fb, k_lens_g)], dim=0)

            cu_q = F.pad(q_lens_g.cumsum(dim=0), (1, 0)).to(torch.int32)
            cu_k = F.pad(k_lens_g.cumsum(dim=0), (1, 0)).to(torch.int32)
            max_q = int(q_lens_g.max().item())
            max_k = int(k_lens_g.max().item())

            hs_flat = flash_attn_varlen_func(q_packed, k_packed, v_packed, cu_q, cu_k, max_q, max_k)
            # hs_flat: (sum_q, H_g, D)

            hs_pad = pad_sequence(
                [hs_flat[s:e] for s, e in zip(cu_q[:-1], cu_q[1:])],
                batch_first=True,
            )  # (B, L_max_q, H_g, D)

            return hs_pad.reshape(B, -1, H_g * head_dim)

        else:
            # SDPA: (B, H_g, Lq, D)
            attn_mask = torch.zeros((B, 1, query_g.size(2), key_g.size(2)),
                                    dtype=torch.bool, device=query_g.device)
            for i, (ql, kl) in enumerate(zip(q_lens_g, k_lens_g)):
                attn_mask[i, :, :ql, :kl] = True

            hs = F.scaled_dot_product_attention(
                query_g, key_g, value_g,
                attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )  # (B, H_g, Lq, D)

            return hs.transpose(1, 2).reshape(B, -1, H_g * head_dim)

    
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        lens=None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)


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
            
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # supporting sequence length
        q_lens = lens.clone() if lens is not None else torch.LongTensor([query.shape[2]] * batch_size).to(query.device)
        k_lens = lens.clone() if lens is not None else torch.LongTensor([key.shape[2]] * batch_size).to(key.device)

        
        # shared attention / temporal attention across batch:
        # assume batch = (target_0, ..., target_{T-1}, reference)
        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1] #Ianna: length of text tokens, originally hardcoded to 512
        else:
            txt_len = 0
        
        """
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

        
        key, value, k_lens = build_shared_kv(key, value, txt_len=txt_len, donors=donors)
        """
        
        # ------------------- Per-head specialization setup -------------------

        Lq_total = query.shape[2]
        assert txt_len <= Lq_total, f"txt_len {txt_len} > seq_len {Lq_total}"
        
        B = key.shape[0]
        ref_count = self._ref_count if self._ref_count is not None else 1
        T = B - ref_count
        ref_indices = list(range(T, B))
        anchor_idx = ref_indices[-1]

        # Head split indices
        H_total = attn.heads
        H_temp = max(1, H_total // 4)
        H_fid = H_total - H_temp

        # Split heads on full sequence first
        q_temp_full = query[:, :H_temp]   # (B, H_temp, L, D)
        k_temp_full = key[:,   :H_temp]
        v_temp_full = value[:, :H_temp]

        q_fid_full  = query[:, H_temp:]   # (B, H_fid, L, D)
        k_fid_full  = key[:,   H_temp:]
        v_fid_full  = value[:, H_temp:]

        
        # donors for fidelity heads: all refs (ref + anchor)
        donors_fid = [[] for _ in range(B)]
        for i in range(T):
            donors = []
            for r in ref_indices:
                donors.append(r)
                if r == anchor_idx and self._anchor_weight > 1.0:
                    extra = int(math.floor(self._anchor_weight - 1.0))
                    donors.extend([anchor_idx] * extra)
            donors_fid[i] = donors
        for r in ref_indices:
            donors_fid[r] = []
        
        # donors for temporal heads: ref + neighbors (override-able by flow)
        flow_fields = self._flow_fields  # optional: (T-1, 2, Hf, Wf)
        donors_temp = [[] for _ in range(B)]
        for i in range(T):
            neighbors = [j for j in range(T) if abs(j - i) <= 5]  # temporal window include self and neighbors
            donors = neighbors + ref_indices
            if self._anchor_weight > 1.0:
                extra = int(math.floor(self._anchor_weight - 1.0))
                donors += [anchor_idx] * extra
            donors_temp[i] = donors
        for r in ref_indices:
            donors_temp[r] = []
        

        # Build K/V separately per head group (this is where routing differs)
        # Expand K/V for image tokens only (txt_len=0 because we already sliced)
        k_fid, v_fid, k_lens_fid = build_shared_kv(k_fid_full, v_fid_full, txt_len=txt_len, donors=donors_fid)
        q_fid = q_fid_full
        q_lens_fid = q_lens
        
        q_temp = q_temp_full[:, :, txt_len:, :]          # (B, H_temp, L_img, D)
        q_lens_temp = (q_lens - txt_len).clamp(min=0)    # valid query lengths for image part

        use_flow = flow_fields is not None
        if use_flow:
            if not isinstance(flow_fields, torch.Tensor):
                flow_fields = torch.tensor(flow_fields)
            use_flow = flow_fields.numel() > 0
        if use_flow:
            # flow-based per-token donors
            flow_fields = flow_fields.to(q_temp.device)
            # L_img should form a square grid
            L_img = q_temp.shape[2]
            H_tok = int(math.sqrt(L_img))
            W_tok = H_tok if H_tok * H_tok == L_img else None
            if W_tok is None:
                use_flow = False
            else:
                # resize flow to token grid and scale displacement to token units
                flow_resized = torch.nn.functional.interpolate(
                    flow_fields, size=(H_tok, H_tok), mode="bilinear", align_corners=False
                )
                if flow_fields.shape[-1] > 0:
                    scale_w = H_tok / flow_fields.shape[-1]
                    scale_h = H_tok / flow_fields.shape[-2]
                else:
                    scale_w = scale_h = 1.0
                flow_resized[:, 0] *= scale_w
                flow_resized[:, 1] *= scale_h

                # precompute grid
                y_coords = torch.arange(H_tok, device=q_temp.device)
                x_coords = torch.arange(H_tok, device=q_temp.device)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
                flat_coords = (grid_y.reshape(-1), grid_x.reshape(-1))

                k_temp_list = []
                v_temp_list = []
                k_lens_temp = []

                for i in range(B):
                    # donors may vary per frame; collect raw then pad
                    if i in ref_indices:
                        donor_frames = [i]
                        donor_idx_list = [flat_coords[0] * W_tok + flat_coords[1]]
                    else:
                        donor_frames = []
                        donor_idx_list = []
                        # self frame (identity)
                        donor_frames.append(i)
                        donor_idx_list.append((grid_y * W_tok + grid_x).reshape(-1))
                        # prev flow
                        if i > 0:
                            flow_prev = flow_resized[i - 1]
                            prev_y = torch.clamp((grid_y - flow_prev[1]).round().long(), 0, H_tok - 1)
                            prev_x = torch.clamp((grid_x - flow_prev[0]).round().long(), 0, H_tok - 1)
                            donor_frames.append(i - 1)
                            donor_idx_list.append((prev_y * W_tok + prev_x).reshape(-1))
                        # next flow
                        if i < T - 1:
                            flow_next = flow_resized[i]
                            next_y = torch.clamp((grid_y + flow_next[1]).round().long(), 0, H_tok - 1)
                            next_x = torch.clamp((grid_x + flow_next[0]).round().long(), 0, H_tok - 1)
                            donor_frames.append(i + 1)
                            donor_idx_list.append((next_y * W_tok + next_x).reshape(-1))
                        # references (same coords), with optional anchor bias
                        for r in ref_indices:
                            donor_frames.append(r)
                            donor_idx_list.append((grid_y * W_tok + grid_x).reshape(-1))
                            if r == anchor_idx and self._anchor_weight > 1.0:
                                extra = int(math.floor(self._anchor_weight - 1.0))
                                for _ in range(extra):
                                    donor_frames.append(anchor_idx)
                                    donor_idx_list.append((grid_y * W_tok + grid_x).reshape(-1))

                    # gather and concat along sequence
                    gathered_k = []
                    gathered_v = []
                    for frm, idxs in zip(donor_frames, donor_idx_list):
                        k_src = k_temp_full[frm, :, txt_len:, :]  # (H_temp, L_img, D)
                        v_src = v_temp_full[frm, :, txt_len:, :]
                        gathered_k.append(k_src[:, idxs, :])
                        gathered_v.append(v_src[:, idxs, :])

                    k_cat = torch.cat(gathered_k, dim=1)  # (H_temp, L_img * n_donors, D)
                    v_cat = torch.cat(gathered_v, dim=1)
                    k_temp_list.append(k_cat)
                    v_temp_list.append(v_cat)
                    k_lens_temp.append(k_cat.shape[1])

                max_Lk = max(k_lens_temp)
                k_temp_padded = []
                v_temp_padded = []
                for k_cat, v_cat in zip(k_temp_list, v_temp_list):
                    if k_cat.shape[1] < max_Lk:
                        pad_len = max_Lk - k_cat.shape[1]
                        k_cat = torch.cat([k_cat, k_cat.new_zeros((k_cat.shape[0], pad_len, k_cat.shape[2]))], dim=1)
                        v_cat = torch.cat([v_cat, v_cat.new_zeros((v_cat.shape[0], pad_len, v_cat.shape[2]))], dim=1)
                    k_temp_padded.append(k_cat)
                    v_temp_padded.append(v_cat)

                k_temp = torch.stack(k_temp_padded, dim=0)  # (B, H_temp, Lk, D)
                v_temp = torch.stack(v_temp_padded, dim=0)
                k_lens_temp = torch.tensor(k_lens_temp, device=k_temp.device, dtype=torch.long)
        if not use_flow:
            k_temp, v_temp, k_lens_temp = build_shared_kv(k_temp_full, v_temp_full, txt_len=txt_len, donors=donors_temp)


        # Run attention twice (memory-efficient kernels)
        hs_temp_img = self._run_attn_group(q_temp, k_temp, v_temp, q_lens_temp, k_lens_temp, attn, head_dim)
        hs_fid_full = self._run_attn_group(q_fid,  k_fid,  v_fid,  q_lens_fid,  k_lens_fid,  attn, head_dim)

        # Concatenate heads back for image tokens
        # Split fidelity output into text/image parts
        hs_fid_txt = hs_fid_full[:, :txt_len, :]     # (B, txt_len, H_fid*D)
        hs_fid_img = hs_fid_full[:, txt_len:, :]     # (B, L_img,  H_fid*D)

        # Build full outputs
        hs_img = torch.cat([hs_temp_img, hs_fid_img], dim=-1)   # (B, L_img, H_total*D)
        
        # Compose text output: must also be (B, txt_len, H_total*D)
        if txt_len > 0:
            pad_txt = hs_fid_txt.new_zeros((B, txt_len, H_temp * head_dim))
            hs_txt  = torch.cat([pad_txt, hs_fid_txt], dim=-1)   # (B, txt_len, H_total*D)
        else:
            hs_txt = hs_img.new_zeros((B, 0, H_total * head_dim))  # (B, 0, 3072)

        hidden_states = torch.cat([hs_txt, hs_img], dim=1)      # (B, L_full, H_total*D)
        hidden_states = hidden_states.to(query.dtype)


        # ------------------- End per-head specialization -------------------
        
        """
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
        """
        
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
