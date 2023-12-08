from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from src.models.tokenizer import AutoencoderConfig, Tokenizer, Encoder, Decoder
from src.default_configs.plater import HydraWrapper


@dataclass
class TokenizerAECfg(HydraWrapper):
    """
    See models.tokenizer.nets.AutoencoderConfig for more details.
    """
    __target__ = AutoencoderConfig
    resolution: int = 64
    in_channels: int = 3
    z_channels: int = 512
    out_channels: int = 3
    channel: int = 64
    channel_scales: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    num_res_blocks: int = 2
    attn_resolutions: List[int] = field(default_factory=lambda: [8, 16])
    dropout: float = 0.0


@dataclass
class TokenizerAE(HydraWrapper):
    __target__ = Encoder
    config: TokenizerAECfg = TokenizerAECfg()


@dataclass
class TokenizerCfg(HydraWrapper):
    """
    See models.tokenizer.tokenizer.Tokenizer for more details.
    """
    __target__ = Tokenizer
    vocab_size: int = 512
    emb_dim: int = 512
    encoder: TokenizerAE = TokenizerAE()
    decoder: TokenizerAE = TokenizerAE()

    def __post_init__(self):
        self.decoder.__init__(**self.encoder.__dict__)
        self.decoder.__target__ = Decoder
