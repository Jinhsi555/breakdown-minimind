import math
import struct
import inspect
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from typing import Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: [batch_size, seq_len, dim]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)  # 输出与输入的数据类型一致，避免后续计算破坏混合精度训练
    
# 在 RoPE 中预先计算旋转角度对应的复数（cosθ + i·sinθ）值 mθ
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # \theta_i = 10000^{-2i/d}, i \in [0, 1, ..., d/2-1]
    t = torch.arange(0, end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # [seq_len, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        # 将 x 的前半部分和后半部分（取反）进行交换，代替 sin 的取反
        return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    # 对 q, k 和 cos, sin 进行广播运算，需要先匹配维度
    # cos, sin [seq_len, head_dim] -> [(1), seq_len, 1, head_dim] 即对所有 batch, head 进行相同的广播运算
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    
    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-5,
        rope_theta: int = 1000000,
        flash_attn: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        
        

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """使 KV 头数适应 Query 头数， 执行矩阵乘法并行运算
    等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # 等价于 x.unsqueeze(3)
        .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    )
    
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads  # query 头映射的 head_dim
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        
    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 接收 cos 和 sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None,):
        batch_size, seq_len, _ = x.shape
        ############## Forward QKV & RoPE ##############
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])  # 截断至 seq_len
        
        # kv_cache 实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 缓存每一个 token 的 k, v
            xv = torch.cat([past_key_value[1], xk], dim=1)
        past_kv = (xk, xv) if use_cache else None
        
        # [batch_size, seq_len, num_heads, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        
        ############ Scaled Dot Production #############
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None  # 这里的 attention_mask 指的是 padding 的掩码
            if attention_mask is not None:
                attn_mask = attention_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.n_local_heads, seq_len, -1)  # attention_mask 形状为 [bsz, seq_len] 扩展后形状为 [bsz, n_heads, seq_len, seq_len]
                attn_mask = attn_mask.bool()
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # 普通注意力机制
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 缩放点积
            scores = scores + torch.triu(
                torch.full((1, 1, seq_len, seq_len), float("inf"), device=scores.device),
                diagonal=1
            )
            
            # 处理 padding 的掩码
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # padding 的部分变为 -inf
                scores += extended_attention_mask
                
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv  # [..., seq_len, seq_len] @ [..., seq_len, head_dim] -> [..., seq_len, head_dim]
        
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)  # -> [batch_size, seq_len, dim] 等价于将所有头的输出维度拼接
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
from transformers.activations import ACT2FN
class FeedForward(nn.Module):
    # input -> RMSNorm ->       up_proj     -> down_proj -> dropout -> output
    #                 \                       /
    #                  -> gate_proj -> SiLU ->
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 向上取整到 64 的倍数
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, hidden_size]
        up_out = self.up_proj(x)
        gate_out = self.gate_proj(x)
        down_out = self.down_proj(self.act_fn(gate_out) * up_out)
        return self.dropout(down_out)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)
        
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)
        
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states  # Multihead Self Attention 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # Feed Forward 残差连接
        return hidden_states, present_key_value

class MiniMindModel(PreTrainedModel):
    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        # 确定当前在序列中的起始位置, 处理增量生成
        # past_key_values[0]：模型第一层的缓存 (K_cache, V_cache)
        # past_key_values[0][0].shape[1] 获取 K_cache 的序列长度
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        hidden_state = self.dropout(self.embed_tokens(input_ids))
        
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_state, present = layer(
                hidden_state,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  # 所有 MultiHead Attention 层的 kv cache
            
        hidden_state = self.norm(hidden_state)
        return hidden_state, presents
    
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):  # 自回归生成函数类
    config_class = MiniMindConfig
    
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重绑定，保证词向量嵌入和输出头的互逆性
        self.OUT = CausalLMOutputWithPast()
        
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_state, past_kvs = self.model(
            input_ids, 
            attention_mask, 
            past_key_values, 
            use_cache,
            **args)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_state[:, slice_indices, :])  # [batch_size, seq_len, hidden_size] 自回归只需要对 hidden_state 最新生成的 token 进行计算即可
        self.OUT.__setitem__('last_hidden_state', hidden_state)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT