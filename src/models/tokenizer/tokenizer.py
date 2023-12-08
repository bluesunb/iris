import torch as th
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange

from src.dataset import Batch
from src.models.tokenizer.lpips import LPIPS
from src.models.tokenizer.nets import Encoder, Decoder
from src.utils import LossWithIntermediateLosses

from typing import Tuple, List, Optional, Union


@dataclass
class TokenizerEncoderOutput:
    """
    Output of the TokenizerEncoder.
    For the shape of output tensors, `t` is optional.

    Attributes:
        z (th.Tensor): size (bs, t, z_ch, h, w), latent vectors that passed conv
        z_quantized (th.Tensor): size (bs, t, emb_dim, h, w), quantized latent vectors
        tokens (th.Tensor): size (bs, t, h * w), index of the nearest embedding vector
    """
    z: th.FloatTensor
    z_quantized: th.FloatTensor
    tokens: th.LongTensor


class Tokenizer(nn.Module):
    """
    Tokenize the input into a sequence of tokens in VQ-VAE manner.
    """

    def __init__(self, vocab_size: int, emb_dim: int, encoder: Encoder, decoder: Decoder, lpips: bool = True):
        """
        Args:
            vocab_size: Size of the vocabulary.
            emb_dim: Dimension of the embedding.
            encoder: Encoder.
            decoder: Decoder.
            lpips: Whether to use LPIPS loss.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.pre_quantize_conv = nn.Conv2d(encoder.config.z_channels, emb_dim, kernel_size=1)
        self.post_quantize_conv = nn.Conv2d(emb_dim, decoder.config.z_channels, kernel_size=1)
        self.lpips = LPIPS().eval() if lpips else None

        self.embedding.weight.data.uniform_(-1 / vocab_size, 1 / vocab_size)

    def __repr__(self):
        return f"Tokenizer(vocab_size={self.vocab_size}, emb_dim={self.embedding.embedding_dim}, " \
               f"encoder={self.encoder}, decoder={self.decoder}, lpips={self.lpips is not None})"

    def encode(self, x: th.Tensor, preprocess: bool = True) -> TokenizerEncoderOutput:
        """
        Encode the input into a sequence of tokens.

        1. Encode the input into a sequence of latent vectors.
            x(bs, t, c, H, W) -> z(bs, t, z_ch, h, w)
        2. Quantize the latent vectors into a sequence of tokens.
            z(bs, t, z_ch, h, w) -> z_quantized(bs, t, emb_dim, h, w) -> tokens(bs, t, h * w)
        
        Args:
            x: (th.Tensor): size (bs, t, c, H, W)
            preprocess: (bool): Whether to preprocess the input.
        
        Returns:
            z, z_quantized, tokens

            - z: (th.Tensor): size (bs, t, z_ch, h, w)
            - z_quantized: (th.Tensor): size (bs, t, emb_dim, h, w)
            - tokens: (th.Tensor): size (bs, t, h * w), index of the nearest embedding vector
        """
        if preprocess:
            x = self.preprocess(x)
        x_shape = x.shape
        x = x.view(-1, *x_shape[-3:])  # (b, c, H, W)   (b = b*t)
        z = self.encoder(x)  # (b, z_ch, h, w)  (h, w = H, W / 2^n)
        z = self.pre_quantize_conv(z)  # (b, e=emb_dim, h, w)
        b, e, h, w = z.shape

        # flatten to compare with codebook
        z_flatten = rearrange(z, 'b e h w -> (b h w) e')  # bhw = num of latent vectors in encoded input `x`,
        # embedding.weight: (v, e)
        distance = th.cdist(z_flatten, self.embedding.weight, p=2) ** 2  # (bhw, v): dist(z_i, e_j)

        tokens = distance.argmin(dim=-1)  # (bhw, ): find the nearest embedding vector idx
        z_quantized = self.embedding(tokens)  # (bhw, emb_dim): look-up

        # restore original shape
        z_quantized = rearrange(z_quantized, '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
        z = z.reshape(*x_shape[:-3], *z.shape[1:])  # (bs, t, e, h, w)
        z_quantized = z_quantized.reshape(*x_shape[:-3], *z_quantized.shape[1:])  # (bs, t, e, h, w)
        tokens = tokens.reshape(*x_shape[:-3], -1)  # (bs, t, h * w)

        return TokenizerEncoderOutput(z, z_quantized, tokens)

    def quantize(self, x: th.Tensor, preprocess: bool = True):
        return self.encode(x, preprocess=preprocess).z_quantized

    def decode(self, z_quantized: th.Tensor, postprocess: bool = False) -> th.Tensor:
        """
        Args:
            z_quantized: (th.Tensor): size (bs, t, emb_dim, h, w)
            postprocess: (bool): Whether to postprocess the output.
        Returns:
            recon
            : (th.Tensor): size (bs, t, c_out, H, W)
        """
        zq_shape = z_quantized.shape
        z_quantized = z_quantized.view(-1, *zq_shape[-3:])  # (b, emb_dim, h, w)
        z_quantized = self.post_quantize_conv(z_quantized)  # (b, z_ch, h, w) -> z_ch: latent vector channel
        recon = self.decoder(z_quantized)  # (b, c_out, H, W)
        recon = recon.reshape(*zq_shape[:-3], *recon.shape[1:])  # (bs, t, c_out, H, W)
        if postprocess:
            recon = self.postprocess(recon)
        return recon

    @th.no_grad()
    def encode_decode(self, x: th.Tensor, preprocess: bool = True, postprocess: bool = False) -> th.Tensor:
        """
        Encode the input into a sequence of tokens and decode the tokens into a sequence of reconstructed images.
        It can warm up the layers.

        Args:
            x (th.Tensor): size (bs, c, H, W)
            preprocess (bool): Whether to preprocess the input.
            postprocess (bool): Whether to postprocess the output.
        """
        z_quantized = self.encode(x, preprocess=preprocess).z_quantized     # (bs, t, emb_dim, h, w)
        return self.decode(z_quantized, postprocess=postprocess)

    def forward(self, x: th.Tensor, preprocess: bool = True, postprocess: bool = False) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor]:
        outputs = self.encode(x, preprocess=preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()  # straight-through estimator
        recon = self.decode(decoder_input, postprocess=postprocess)
        return outputs.z, outputs.z_quantized, recon

    def compute_loss(self, batch: Batch, **kwargs) -> LossWithIntermediateLosses:
        assert self.lpips is not None, "LPIPS is not initialized."
        obs = rearrange(batch["observations"], 'b t c h w -> (b t) c h w')
        obs = self.preprocess(obs)
        z, z_quantized, recon = self.forward(obs, preprocess=False, postprocess=False)

        beta = kwargs.get("beta", 1.0)
        vq_loss = (z.detach() - z_quantized).pow(2).mean()
        commitment_loss = beta * (z - z_quantized.detach()).pow(2).mean()
        recon_loss = th.abs(obs - recon).mean()
        perceptual_loss = th.mean(self.lpips(obs, recon))

        return LossWithIntermediateLosses(vq_loss=vq_loss,
                                          commitment_loss=commitment_loss,
                                          recon_loss=recon_loss,
                                          perceptual_loss=perceptual_loss)

    def preprocess(self, x: th.Tensor) -> th.Tensor:
        """
        map x (0 <= x <= 1) to (-1 <= x <= 1)
        """
        return x.mul(2).sub(1)  # -1 <= x <= 1

    def postprocess(self, y: th.Tensor) -> th.Tensor:
        """
        map y (-1 <= y <= 1) to (0 <= y <= 1)
        """
        return y.add(1).div(2)

    def __repr__(self):
        return f"Tokenizer(vocab_size={self.vocab_size}, emb_dim={self.embedding.embedding_dim})"
    
    def __str__(self):
        return super().__repr__()


if __name__ == "__main__":
    from src.models.tokenizer.nets import AutoencoderConfig

    config = AutoencoderConfig(resolution=64,
                               in_channels=3,
                               z_channels=128,
                               out_channels=24,
                               channel=32,
                               channel_scales=(2, 3, 4),
                               num_res_blocks=2,
                               attn_resolutions=(128,),
                               dropout=0.5)

    encoder = Encoder(config)
    decoder = Decoder(config)
    tokenizer = Tokenizer(100, 144, encoder, decoder, lpips=False)

    batch_size, seq_len = 3, 5
    x = th.randn(batch_size, seq_len, 3, 64, 64)
    z, z_quantized, recon = tokenizer(x)
    print(z.shape)
    print(z_quantized.shape)
    print(recon.shape)
