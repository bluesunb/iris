from dataclasses import dataclass
from src.default_configs.plater import HydraWrapper
from src.models.transformer import TransformerConfig


@dataclass
class WorldModelCfg(HydraWrapper):
    """
    See models.transformer.TransformerConfig for more details.
    This is for the world model and its transformer model.
    """
    __target__ = TransformerConfig
    tokens_per_block: int = 17
    max_blocks: int = 20
    attention: str = "causal"
    n_layers: int = 10
    n_heads: int = 4
    emb_dim: int = 256      # = transformer hidden size
    emb_dropout: float = 0.1
    residual_dropout: float = 0.1
    attn_dropout: float = 0.1
