import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from einops import rearrange

# import sys
# sys.path.append('/home/bluesun/PycharmProjects/research/iris')

from src.models.kv_caching import KVCache, KeysValues
from typing import Optional


@dataclass
class TransformerConfig:
    """
    Transformer configuration.

    Notes:
        - `block` means the single (s,a) token which consists of `tokens_per_block - 1` obs tokens and 1 action token.
        - `tokens_per_block` and `max_blocks` are used to calculate `max_tokens`.

    Args:
        tokens_per_block (int): Number of tokens (tok_obs + tok_act) per block.
        max_blocks (int): Maximum number of blocks.
        attention (str): Attention type. 'causal' or 'block_causal'.
        n_layers (int): Number of transformer block layers.
        n_heads (int): Number of attention heads.
        emb_dim (int): Embedding dimension over all heads.
        emb_dropout (float): Dropout rate for embedding layer.
        residual_dropout (float): Dropout rate for residual connection.
        attn_dropout (float): Dropout rate for attention layer.
    """
    tokens_per_block: int
    max_blocks: int
    attention: str
    n_layers: int
    n_heads: int
    emb_dim: int
    emb_dropout: float = 0.0
    residual_dropout: float = 0.0
    attn_dropout: float = 0.0
    
    @property
    def max_tokens(self) -> int:
        return self.tokens_per_block * self.max_blocks
    

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.emb_dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.emb_dim)

    def generate_empty_kvs(self, n_keys: int, max_tokens: int) -> KeysValues:
        device = self.layer_norm.weight.device
        return KeysValues(n_tokens=n_keys,
                          n_heads=self.config.n_heads,
                          max_tokens=max_tokens,
                          emb_dim=self.config.emb_dim,
                          num_layers=self.config.n_layers,
                          device=device)
    
    def forward(self, sequences: th.Tensor, prev_kvs: Optional[KeysValues] = None) -> th.Tensor:
        assert prev_kvs is None or len(prev_kvs) == len(self.blocks)
        x = self.dropout(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if prev_kvs is None else prev_kvs[i])
        
        x = self.layer_norm(x)
        return x

    def __repr__(self):
        return f"Transformer(emb_dim={self.config.emb_dim}, num_layers={self.config.n_layers})"

    def str(self):
        return super().__repr__()


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.emb_dim)
        self.layer_norm2 = nn.LayerNorm(config.emb_dim)
        self.attn = SelfAttention(config)
        emb_dim = config.emb_dim
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim),
                                 nn.GELU(),
                                 nn.Linear(4 * emb_dim, emb_dim),
                                 nn.Dropout(config.residual_dropout))
        
    def forward(self, x: th.Tensor, prev_kvs: Optional[KeysValues] = None) -> th.Tensor:
        x_attn = self.attn(self.layer_norm1(x), prev_kvs)
        x = x + x_attn
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0, "emb_dim must be divisible by n_heads"
        assert config.attention in ('causal', 'block_causal'), "attention must be 'causal' or 'block_causal'"
        self.n_heads = config.n_heads

        self.Q = nn.Linear(config.emb_dim, config.emb_dim)
        self.K = nn.Linear(config.emb_dim, config.emb_dim)
        self.V = nn.Linear(config.emb_dim, config.emb_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)

        causal_mask = th.tril(th.ones(config.max_tokens, config.max_tokens))  # lower triangular matrix

        diag_blocks = [th.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]
        block_causal_mask = th.max(causal_mask,
                                   th.block_diag(*diag_blocks))  # block diagonal matrix with causal mask on each block

        self.register_buffer("mask", causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: th.Tensor, kv_cache: Optional[KVCache] = None) -> th.Tensor:
        bs, t, emb_dim = x.shape
        cache_size = 0
        if kv_cache is not None:    # check cache shape
            b, n_heads, cache_size, head_dim = kv_cache.shape
            assert n_heads == self.n_heads and b == bs and head_dim * n_heads == emb_dim, \
                f"({b, n_heads, n_heads * head_dim}) != ({bs, self.n_heads, emb_dim})"

        head_dim = emb_dim // self.n_heads
        q = self.Q(x).view(bs, t, self.n_heads, head_dim).transpose(1, 2)
        k = self.K(x).view(bs, t, self.n_heads, head_dim).transpose(1, 2)
        v = self.V(x).view(bs, t, self.n_heads, head_dim).transpose(1, 2)

        if kv_cache is not None:
            kv_cache.update(k, v)   # store k, v to the cache
            k, v = kv_cache.get()   # get k, v from the cache

        attn = th.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(k.shape[-1])  # (bs, n_heads, t, t)
        attn = attn.masked_fill(self.mask[cache_size:cache_size + t, :cache_size + t] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v    # (bs, n_heads, t, head_dim)
        y = rearrange(y, 'b h t d -> b t (h d)')    # (bs, t, emb_dim)
        y = self.residual_dropout(self.proj(y))
        return y


if __name__ == '__main__':
    from src.models.tokenizer import AutoencoderConfig, Encoder, Decoder, Tokenizer
    
    ae_config = AutoencoderConfig(resolution=64,
                                  in_channels=3,
                                  z_channels=128,
                                  out_channels=24,
                                  channel=32,
                                  channel_scales=(2, 3, 4),
                                  num_res_blocks=2,
                                  attn_resolutions=(128,),
                                    dropout=0.5)
    
    encoder = Encoder(ae_config)
    decoder = Decoder(ae_config)
    vocab_size = 100
    emb_dim = 144
    tokenizer = Tokenizer(vocab_size=vocab_size,
                          emb_dim=emb_dim,
                          encoder=encoder,
                          decoder=decoder,
                          lpips=False)

    batch_size = 2
    n_tokens = 11
    x = th.randn(batch_size, n_tokens, ae_config.in_channels, ae_config.resolution, ae_config.resolution)
    tokenizer_output = tokenizer.encode(x, preprocess=True)
    print(tokenizer_output.z.shape)
    print(tokenizer_output.z_quantized.shape)
    print(tokenizer_output.tokens.shape)

    tf_config = TransformerConfig(tokens_per_block=16,
                                  max_blocks=4,
                                  attention='block_causal',
                                  n_layers=2,
                                  n_heads=3,
                                  emb_dim=15,
                                  emb_dropout=0.0,
                                  residual_dropout=0.0,
                                  attn_dropout=0.0)

    transformer = Transformer(tf_config)
    prev_kv = KeysValues(n_tokens=n_tokens,
                         n_heads=tf_config.n_heads,
                         max_tokens=tf_config.max_tokens,
                         emb_dim=emb_dim,
                         num_layers=tf_config.n_layers,
                         device=th.device('cpu'))
    
    out = transformer(tokenizer_output.tokens, prev_kv)
    print(out.shape)
