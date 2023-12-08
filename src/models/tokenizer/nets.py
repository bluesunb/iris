import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Union


def normalize(in_channels: int) -> nn.Module:
    """
    Group normalization = Layer normalization with grouping along the channel dimension. (dim=1)
    """
    # return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    # return nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)
    return nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x: th.Tensor) -> th.Tensor:
    """
    Swish activation function.
    """
    return x * th.sigmoid(x)


@dataclass
class AutoencoderConfig:
    """
    Args:
        resolution (int): The resolution of the input image.
        in_channels (int): The number of input channels.
        z_channels (int): The number of channels of the latent vector (encoder output).
        out_channels (int): The number of output channels.
        channel (int): Unit channel size.
        channel_scales (List[int]): The normalized output channels of each resolution. (divided by `channel`)
        num_res_blocks (int): The number of residual blocks in each resolution.
        attn_resolutions (List[int]): The resolutions to apply attention blocks.
        dropout (float): The dropout rate.
    """
    resolution: int
    in_channels: int
    z_channels: int
    out_channels: int
    channel: int
    channel_scales: List[int]  # normalized output channels of each resolution
    num_res_blocks: int
    attn_resolutions: List[int]
    dropout: float


class SamplerBlock(nn.Module):
    """
    Block to apply up/down sampling on raw image or feature map.

    Attributes:
        res_blocks: (nn.ModuleList[ResnetBlock]): The list of residual blocks to extract features.
        attn_blocks: (nn.ModuleList[AttentionBlock] | Empty):
            The list of attention blocks to apply attention along with a feature extraction.
        sampler: (nn.Identity | Downsample | Upsample): The sampler to apply up/down sampling.
    """
    def __init__(self, upsample: bool = False):
        super().__init__()
        self.res_blocks: nn.ModuleList[ResnetBlock] = nn.ModuleList()
        self.attn_blocks: nn.ModuleList[AttentionBlock] = nn.ModuleList()
        self.sampler: Union[nn.Identity, Downsample, Upsample] = nn.Identity()

    def forward(self, x: th.Tensor, timestep_emb: Optional[th.Tensor], sample: bool = False) -> th.Tensor:
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x, timestep_emb)
            if len(self.attn_blocks) > 0:
                x = self.attn_blocks[i](x)
        if sample:
            x = self.sampler(x)
        return x


class MidBlock(nn.Module):
    """
    Middle block of the encoder and decoder.
    """
    def __init__(self):
        super().__init__()
        self.block1: Union[nn.Identity, ResnetBlock] = nn.Identity()
        self.attn1: Union[nn.Identity, AttentionBlock] = nn.Identity()
        self.block2: Union[nn.Identity, ResnetBlock] = nn.Identity()

    def forward(self, x: th.Tensor, timestep_emb: Optional[th.Tensor]) -> th.Tensor:
        h = self.block1(x, timestep_emb)
        h = self.attn1(h)
        h = self.block2(h, timestep_emb)
        return h


class Encoder(nn.Module):
    """
    Encode the input tensor into latent vector.

    Attributes:
        config: (AutoencoderConfig): The configuration of the encoder.
        num_resolutions: (int): The number of different resolutions to


    """
    def __init__(self, config: AutoencoderConfig):
        """
        Args:
            config: Encoder configuration.
        """
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.channel_scales)
        timestep_emb_channels = 0

        # down sampling
        self.conv_in = nn.Conv2d(config.in_channels, config.channel, kernel_size=3, stride=1, padding=1)
        self.conv_downs = self.construct_downsampler(timestep_emb_channels)

        block_in = config.channel * config.channel_scales[-1]  # last block_in for `construct_downsampler
        self.mid = self.construct_middle(in_channels=block_in, timestep_emb_channels=timestep_emb_channels)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, config.z_channels, kernel_size=3, stride=1, padding=1)

    def construct_downsampler(self, timestep_emb_channels: int) -> "nn.ModuleList[SamplerBlock]":
        """
        Constructs downsampling layers.

        Downsample layer:
            nn.ModuleList([
                nn.ModuleList([ResnetBlock, ResnetBlock, ... AttentionBlock, ResnetBlock, ...]),
                ...
                nn.ModuleList([ResnetBlock, ResnetBlock, ... Downsample]),
            ])

        Args:
            timestep_emb_channels (int): The number of channels of timestep embedding.
        """
        current_resolution = self.config.resolution
        conv_downs = nn.ModuleList()
        # if `channel_scales` is (1, 2, 4, 8), then it use `in_channels` as channel * channel_scales[i]
        in_channel_scales = (1,) + tuple(self.config.channel_scales)
        for n in range(self.num_resolutions):
            down_block = SamplerBlock()
            block_in = self.config.channel * in_channel_scales[n]
            block_out = self.config.channel * self.config.channel_scales[n]
            for i in range(self.config.num_res_blocks):
                # Stack residual blocks and if current resolution is in `attn_resolutions`,
                # then stack attention blocks.
                down_block.res_blocks.append(ResnetBlock(in_channels=block_in,
                                                         out_channels=block_out,
                                                         timestep_emb_channels=timestep_emb_channels,
                                                         dropout=self.config.dropout))
                block_in = block_out
                if current_resolution in self.config.attn_resolutions:
                    down_block.attn_blocks.append(AttentionBlock(block_in))

            if n != self.num_resolutions - 1:  # except for last
                down_block.sampler = Downsample(block_in, with_conv=True)
                current_resolution //= 2
            conv_downs.append(down_block)

        return conv_downs

    def construct_middle(self, in_channels: int, timestep_emb_channels: int) -> MidBlock:
        mid = MidBlock()
        mid.block1 = ResnetBlock(in_channels=in_channels,
                                 out_channels=in_channels,
                                 timestep_emb_channels=timestep_emb_channels,
                                 dropout=self.config.dropout)
        mid.attn1 = AttentionBlock(in_channels)
        mid.block2 = ResnetBlock(in_channels=in_channels,
                                 out_channels=in_channels,
                                 timestep_emb_channels=timestep_emb_channels,
                                 dropout=self.config.dropout)
        return mid

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Apply residual block + attention block + downsample + middle block.

        Args:
            x: (bs, c, h, w): The input tensor.
        Returns:
            (th.Tensor): (bs, z_c, h//2^n, w//2^n): The output tensor.

        """
        timestep_emb = None

        z = self.conv_in(x)     # (bs, z_c, h, w)
        for n in range(self.num_resolutions):
            z = self.conv_downs[n](z, timestep_emb, sample=(n != self.num_resolutions - 1))
        z = self.mid(z, timestep_emb)
        z = swish(self.norm_out(z))
        z = self.conv_out(z)  # (bs, z_c, h//2^n, w//2^n)
        return z
    
    def __repr__(self):
        s = ', '.join(f'{k}={v}' for k, v, in self.config.__dict__.items())
        return f"Encoder(" + s + ")"

    def __str__(self):
        return super().__repr__()


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.channel_scales)
        timestep_emb_channels = 0

        block_in = config.channel * config.channel_scales[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(config.z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = self.construct_middle(in_channels=block_in, timestep_emb_channels=timestep_emb_channels)
        self.conv_ups = self.construct_upsampler(timestep_emb_channels)

        block_in = config.channel * config.channel_scales[0]
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, config.out_channels, kernel_size=3, stride=1, padding=1)

    def construct_middle(self, in_channels: int, timestep_emb_channels: int) -> MidBlock:
        mid = MidBlock()
        mid.block1 = ResnetBlock(in_channels, in_channels,
                                 timestep_emb_channels=timestep_emb_channels,
                                 dropout=self.config.dropout)
        mid.attn1 = AttentionBlock(in_channels)
        mid.block2 = ResnetBlock(in_channels, in_channels,
                                 timestep_emb_channels=timestep_emb_channels,
                                 dropout=self.config.dropout)

        return mid

    def construct_upsampler(self, timestep_emb_channels: int) -> nn.ModuleList:
        block_in = self.config.channel * self.config.channel_scales[self.num_resolutions - 1]
        current_resolution = self.config.resolution // (2 ** (self.num_resolutions - 1))
        print(f"Decoder : shape of latent vector: {self.config.z_channels, current_resolution, current_resolution}")

        conv_ups = []
        for n in reversed(range(self.num_resolutions)):
            up_block = SamplerBlock()
            block_out = self.config.channel * self.config.channel_scales[n]
            for i in range(self.config.num_res_blocks + 1):
                up_block.res_blocks.append(ResnetBlock(in_channels=block_in,
                                                       out_channels=block_out,
                                                       timestep_emb_channels=timestep_emb_channels,
                                                       dropout=self.config.dropout))
                block_in = block_out
                if current_resolution in self.config.attn_resolutions:
                    up_block.attn_blocks.append(AttentionBlock(block_in))

            if n != 0:
                up_block.sampler = Upsample(block_in, with_conv=True)
                current_resolution *= 2
            conv_ups.append(up_block)
        conv_ups = nn.ModuleList(conv_ups[::-1])
        return conv_ups

    def forward(self, z: th.Tensor) -> th.Tensor:
        timestep_emb = None

        h = self.conv_in(z)
        h = self.mid(h, timestep_emb)

        #upsampling
        for n in reversed(range(self.num_resolutions)):
            h = self.conv_ups[n](h, timestep_emb, sample=(n != 0))
        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return h
    
    def __repr__(self):
        s = ', '.join(f'{k}={v}' for k, v, in self.config.__dict__.items())
        return f"Decoder(" + s + ")"

    def __str__(self):
        return super().__repr__()


class Upsample(nn.Module):
    """
    Upsamples the input by a factor of 2 with the nearest interpolation.
    If `with_conv` is True, then it applies a 3x3 convolution on the upsampled input.
    """

    def __init__(self, in_channels: int, with_conv: bool):
        """
        Args:
            in_channels (int): The number of input channels.
            with_conv (bool): Whether to use convolution at last to postprocess the interpolated input.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # (c, h, w) -> (c, h, w)
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')  # (c, h, w) -> (c, 2h, 2w)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsamples the input by a factor of 2 with average pooling or convolution.
    If `with_conv` is True, convolution is used.
    """

    def __init__(self, in_channels: int, with_conv: bool):
        """
        Args:
            in_channels (int): The number of input channels.
            with_conv (bool): Whether to use convolution for downsampling.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # (c, h, w) -> (c, h/2, w/2)
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)  # pad at the bottom and right
            x = F.pad(x, pad, mode="constant", value=0)  # (c, h, w) -> (c, h+1, w+1)
            x = self.conv(x)  # (c, h+1, w+1) -> (c, h/2, w/2)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)  # (c, h, w) -> (c, h/2, w/2)
        return x


class ResnetBlock(nn.Module):
    """
    A residual block with skip connections. There are also timestep embeddings.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        use_conv_shortcut (bool): Whether to use large size kernel for the convolutional shortcut.
        shortcut: The shortcut connection. Through this, the input tensor skip the blocks.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_shortcut: bool = False,
                 timestep_emb_channels: int = 512,
                 dropout: float = 0.0):
        """
        - residual block: {norm, swish, (dropout), conv} x 2

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            conv_shortcut (bool): Whether to use large size kernel for the convolutional shortcut.
            dropout (float): The dropout rate.
            timestep_emb_channels (int): The number of channels of timestep embedding.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        out_channels = self.out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if timestep_emb_channels > 0:
            self.timestep_proj = nn.Linear(timestep_emb_channels, self.out_channels)

        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # residual shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: th.Tensor, timestep_emb: th.Tensor) -> th.Tensor:
        """
        Args:
            x (th.Tensor): (bs, c, h, w), The input tensor.
            timestep_emb (th.Tensor): (bs, t_c), The timestep embedding tensor.

        Returns:
            (th.Tensor): (bs, c, h, w), The output tensor.
        """
        h = x
        # block 1
        for layer in [self.norm1, swish, self.conv1]:
            h = layer(h)  # (bs, out_c, h, w)

        # timestep_emb: (bs, t_c)
        if timestep_emb is not None:
            h = h + self.timestep_proj(timestep_emb)[:, :, None, None]

            # block 2
        for layer in [self.norm2, swish, self.dropout, self.conv2]:
            h = layer(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    """
    Attention block with skip connection.
    All the convolutions are 1x1 convolutions so that the spatial resolution is not changed.

    Attributes:
        in_channels (int): The number of input channels.
        norm: The normalization layer.
        Q: The query projection layer.
        K: The key projection layer.
        V: The value projection layer.
        proj_out: The output projection layer.
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels (int): The number of input channels, which is the in_channels of the residual block.
        """
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Apply self attention on x with residual connection.

        Args:
            x: (bs, c, h, w): The input tensor.
        Returns:
            (th.Tensor): (bs, c, h, w): The output tensor.
        """
        z = x

        # Compute Attention
        z = self.norm(z)
        q, k, v = self.Q(z), self.K(z), self.V(z)
        q = q.flatten(2).transpose(1, 2)  # (bs, c, h, w) -> (bs, c, h*w) -> (bs, h*w, c)
        k = k.flatten(2)  # (bs, c, h, w) -> (bs, c, h*w)
        attn_weight = th.bmm(q, k) / math.sqrt(q.size(2))  # (bs, h*w, h*w)
        attn_weight = F.softmax(attn_weight, dim=-1)

        # Apply Attention
        v = v.flatten(2).transpose(1, 2)  # (bs, c, h, w) -> (bs, c, h*w) -> (bs, h*w, c)
        z = th.bmm(attn_weight, v)  # (bs, h*w, c)
        z = z.transpose(1, 2).reshape_as(x)  # (bs, c, h, w)
        z = self.proj_out(z)

        return x + z


if __name__ == '__main__':
    config = AutoencoderConfig(resolution=64,
                               in_channels=3,
                               z_channels=129,
                               out_channels=3,
                               channel=16,
                               channel_scales=[1,2,3,4],
                               num_res_blocks=2,
                               attn_resolutions=[16, 64],
                               dropout=0.0)
    
    def encoder_init(encoder: Encoder):
        for name, param in encoder.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0.0, 0.02)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def num_of_params(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    from pytorch_lightning import seed_everything
    seed_everything(42)
    encoder = Encoder(config)
    decoder = Decoder(config)

    print(f"# of param in Encoder: {num_of_params(encoder)}")
    print(f"# of param in Decoder: {num_of_params(decoder)}")

    seed_everything(42)
    encoder_init(encoder)

    z = th.full((1, 129, 8, 8), 0.5)
    x_hat = decoder.forward(z)
    print(x_hat.mean(), x_hat.std())

    import time
    elapsed = 0
    for i in range(30):
        x = th.randn(1, 3, 64, 64)
        start = time.time()
        z = encoder.forward(x)
        elapsed += time.time() - start

    print(f"Elapsed time: {elapsed/30:.4f} sec")
    
    print(z.mean(), z.std())

    # x_hat = decoder(z)
    # print(x.shape, z.shape, x_hat.shape)