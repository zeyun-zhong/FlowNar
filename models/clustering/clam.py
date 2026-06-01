"""
Author: Zeyun Zhong
Email: zeyun.zhong@kit.edu
"""

from typing import TYPE_CHECKING, Optional, Tuple, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from models.clustering.layernorm import RMSNorm
from models.clustering.activations import swiglu_linear
import copy

from models.clustering.scan import sequential_scan


@dataclass
class CLAMInferenceParams:
    num_layers: int
    states: List[Optional[Tensor]] = field(init=False)
    self_attn_out_first_layer: Tensor = None
    q_first_layer: Tensor = None
    g_first_layer: Tensor = None

    def __post_init__(self):
        self.states = [None] * self.num_layers


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GatedCrossLinearAttention(nn.Module):
    def __init__(self, d_model, nhead, gate_low_rank_dim=16, expand_k=0.5,
                 gate_state=True, use_scan=True, **kwargs):
        super().__init__()
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        qk_dim = int(d_model * expand_k)
        self.scale = (qk_dim // nhead) ** -0.5
        self.q = nn.Linear(d_model, qk_dim, bias=False)
        self.k = nn.Linear(d_model, qk_dim, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)

        self.gate_state = gate_state
        if gate_state:
            self.gk_proj = nn.Sequential(
                nn.Linear(d_model, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, qk_dim, bias=True)
            )
        else:
            self.gk_proj = nn.Linear(d_model, self.num_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_norm = RMSNorm(self.head_dim)
        self.gate_fn = nn.SiLU()
        self.gate_logit_normalizer = 16
        self.use_scan = use_scan

    def forward(self, query, key, value,
                inference_params: CLAMInferenceParams = None,
                layer_idx=0, *args, **kwargs):
        # Streaming inference
        if inference_params is not None:
            return self.forward_one_step(query, key, value, inference_params, layer_idx)

        q = self.q(query)  # b n d/2
        k = self.k(key)
        v = self.v(value)
        gk = self.gk_proj(value)  # state gate
        g = self.g_proj(query)  # output gate

        q, k, v, gk = (
            rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads) for x in
            (q, k, v, gk))

        q = q * self.scale

        kv = torch.einsum("bhlm,bhld->bhlmd", k, v)  # b h l d/2 d

        if self.use_scan:
            return self.forward_all_steps_scan(q, kv, gk, g)
            # return self.forward_all_steps(q, kv, gk, g)

        # Applying gate for s
        if self.gate_state:
            gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # b h l d/2
            gk = torch.cumsum(gk, dim=2).flip(2).unsqueeze(-1)  # b h l d/2 1 (in log space)
            gk = torch.exp(gk)  # to range (0, 1)
        else:
            # Applying gate for kv
            gk = F.sigmoid(gk).unsqueeze(-1)

        s = (gk * kv).sum(2) # b h d/2 d
        o = q @ s  # b h n d
        o_ = self.g_norm(o)
        o = rearrange(o_, 'b h l d -> b l (h d)')
        o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o, None

    def forward_all_steps(self, q, kv, gk, g):
        b, h, l, c, d = kv.shape
        gk = F.sigmoid(gk).unsqueeze(-1)  # b h l c 1
        s = torch.zeros(b, h, c, d, dtype=kv.dtype, device=kv.device)
        states = torch.empty(b, h, l, c, d, dtype=kv.dtype, device=kv.device)
        for i in range(l):
            s = gk[:, :, i] * s + kv[:, :, i]
            states[:, :, i] = s
        q = q.unsqueeze(2).expand(-1, -1, l, -1, -1)  # b h l n c
        o = q @ states  # b h l n d
        o = rearrange(self.g_norm(o), 'b h l n d -> b l n (h d)')
        g = g.unsqueeze(1).expand(-1, l, -1, -1)  # b l n d
        o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o, None

    def forward_all_steps_scan(self, q, kv, gk, g):
        b, h, l, c, d = kv.shape
        gk = F.sigmoid(gk)  # b h l c
        states = sequential_scan(gk, kv)
        q = q.unsqueeze(2).expand(-1, -1, l, -1, -1)  # b h l n c
        o = q @ states  # b h l n d
        o = rearrange(self.g_norm(o), 'b h l n d -> b l n (h d)')
        g = g.unsqueeze(1).expand(-1, l, -1, -1)  # b l n d
        o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o, None

    def forward_one_step(
            self, query, key, value,
            inference_params: CLAMInferenceParams, layer_idx: int
    ):
        assert inference_params is not None
        # Must be calculated
        k = self.k(key)
        v = self.v(value)
        gk = self.gk_proj(value)  # state gate

        k, v, gk = (
            rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads) for x in
            (k, v, gk))

        kv = torch.einsum("bhlm,bhld->bhlmd", k, v)  # b h l d/2 d

        # Get q and g
        if layer_idx == 0 and inference_params.q_first_layer is not None:
            q = inference_params.q_first_layer
            gated_g = inference_params.g_first_layer
        else:
            q = self.q(query)  # b n d/2
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            q = q * self.scale
            g = self.g_proj(query)  # output gate
            gated_g = self.gate_fn(g)

            if layer_idx == 0:
                inference_params.q_first_layer = q
                inference_params.g_first_layer = gated_g

        b, h, l, c, d = kv.shape
        # Get state
        if inference_params.states[layer_idx] is not None:
            state = inference_params.states[layer_idx]
        else:
            state = torch.zeros(b, h, c, d, dtype=kv.dtype, device=kv.device)  # b h d/2 d

        # Iterate over temporal length (can be accelerated via scan)
        gk = F.sigmoid(gk).unsqueeze(-1)
        for i in range(l):
            state = gk[:, :, i] * state + kv[:, :, i]
        # Cache state
        inference_params.states[layer_idx] = state

        o = q @ state  # b h n d
        o = rearrange(self.g_norm(o), 'b h n d -> b n (h d)')
        o = o * gated_g
        o = self.o_proj(o)
        return o, state


class GLAMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)


def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")


class GLADecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True, gate_state=True):
        super().__init__()

        # For Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)

        # For Cross-Attention
        self.norm2 = RMSNorm(d_model)
        self.multihead_attn = GatedCrossLinearAttention(d_model, nhead, gate_state=gate_state)

        self.mlp_norm = RMSNorm(d_model)
        self.mlp = GLAMLP(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                inference_params: CLAMInferenceParams = None,
                layer_idx: int = 0):
        # Self-attention
        if layer_idx == 0 and (inference_params is not None) and (inference_params.self_attn_out_first_layer is not None):
            tgt = inference_params.self_attn_out_first_layer
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            if inference_params is not None and layer_idx == 0:
                inference_params.self_attn_out_first_layer = tgt

        # Cross Linear Attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            inference_params=inference_params,
            layer_idx=layer_idx,
        )[0]

        if tgt2.dim() > tgt.dim() :
            tgt = tgt.unsqueeze(1).expand(-1, memory.size(1), -1, -1)

        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.mlp(tgt)
        tgt = tgt + tgt2
        tgt = self.mlp_norm(tgt)
        return tgt


class CLAM(nn.Module):
    def __init__(self, d_model, n_head=8, n_clusters=10, dropout=0.1, learnable_tgt=False, n_layers=1):
        super().__init__()

        decoder_layer = GLADecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            gate_state=True,
        )

        self.n_layers = n_layers
        self.layers = _get_clones(decoder_layer, n_layers)
        self.learnable_tgt = learnable_tgt
        prenorm = False
        self.norm = nn.LayerNorm(d_model) if prenorm else nn.Identity()

        if learnable_tgt:
            self.tgt_embed = nn.Embedding(n_clusters, d_model)

        self.query_embed = nn.Embedding(n_clusters, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_inference_params(self):
        return CLAMInferenceParams(num_layers=self.n_layers)

    def forward(self, memory, tgt_mask=None, memory_padding_mask=None,
                tgt_padding_mask=None, memory_pos=None, inference_params=None):
        B, T, _ = memory.shape
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        if self.learnable_tgt:
            tgt = self.tgt_embed.weight.unsqueeze(0).expand(B, -1, -1)
        else:
            tgt = torch.zeros_like(query)


        out = self._forward(
            tgt, memory, tgt_mask, query_pos=query,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask, pos=memory_pos,
            inference_params=inference_params,
        )
        return out

    def _forward(self, tgt, memory,
                 tgt_mask: Optional[Tensor] = None,
                 memory_mask: Optional[Tensor] = None,
                 tgt_key_padding_mask: Optional[Tensor] = None,
                 memory_key_padding_mask: Optional[Tensor] = None,
                 pos: Optional[Tensor] = None,
                 query_pos: Optional[Tensor] = None,
                 inference_params: CLAMInferenceParams = None):
        output = tgt

        for i, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           inference_params=inference_params,
                           layer_idx=i)

        output = self.norm(output)

        return output