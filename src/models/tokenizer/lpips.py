import os, requests, hashlib
import torch as th
import torch.nn as nn
from torchvision import models

from functools import reduce
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm

from typing import List, Optional, Tuple, Union, NamedTuple


URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}

VggOutputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])


def normalize_tensor(x: th.Tensor, eps: float = 1e-10) -> th.Tensor:
    rms = th.sqrt(th.mean(x ** 2, dim=1, keepdim=True))
    return x / (rms + eps)


def spatial_average(x: th.Tensor, keepdim: bool = True) -> th.Tensor:
    """
    Average tensor along spatial dimensions.
    Args:
        x: (bs, c, h, w): The input tensor.
        keepdim: (bool): Whether to keep the dimensions or not.
    Returns:
        (bs, c, 1, 1): The averaged tensor.
    """
    return x.mean(dim=[2, 3], keepdim=keepdim)


class VGG16(nn.Module):
    def __init__(self, pretrained: bool = True, requires_grad: bool = True):
        super().__init__()
        # vgg_pretrained_features = models.vgg16(pretrained=pretrained).features  # feature extractor of vgg16
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features  # feature extractor of vgg16
        self.slice1 = vgg_pretrained_features[:4]  # Conv2d ~ ReLU
        self.slice2 = vgg_pretrained_features[4:9]  # MaxPool2d ~ ReLU
        self.slice3 = vgg_pretrained_features[9:16]  # MaxPool2d ~ ReLU
        self.slice4 = vgg_pretrained_features[16:23]  # MaxPool2d ~ ReLU
        self.slice5 = vgg_pretrained_features[23:30]  # MaxPool2d ~ ReLU
        self.n_slices = 5

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: th.Tensor) -> "NamedTuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]":
        """
        Forward pass through the VGG16 network.
        It returns the outputs of the intermediate layers of the network.
        Args:
            x: (th.Tensor): The input tensor.

        Returns:
            The outputs of the intermediate layers. It formed as namedtuple: VggOutputs.
            VggOutputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        """
        outs = []
        h = x
        for i in range(self.n_slices):
            h = getattr(self, f'slice{i + 1}')(h)
            outs.append(h)
        return VggOutputs(*outs)
    
    def __repr__(self):
        return f"VGG16"
    
    def __str__(self):
        return super().__repr__()


class ScalingLayer(nn.Module):
    """
    A layer that scales the input to the range [-1, 1].
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('shift', th.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]) # (1, 3, 1, 1)
        self.register_buffer('scale', th.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return (x - self.shift) / self.scale


class SingleConv(nn.Module):
    """
    A single 1-kernel size convolutional layer with optional dropout.
    """

    def __init__(self, in_channels: int, out_channels: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
        if use_dropout:
            layers.insert(0, nn.Dropout2d(p=0.5))
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    This metric is used to measure the similarity between two images by comparing the intermediate features of output
    of a pretrained VGG16 network.

    Attributes:
        scaling_layer (ScalingLayer): A layer that scales the input to the range [-1, 1].
        channels (List[int]): The number of channels of the intermediate layers of the VGG16 network.
        net (VGG16): The pretrained VGG16 network.
        conv{i} (SingleConv): A single convolutional layer to convert the intermediate features of the VGG16 network to
            a single channel.
    """
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]  # vgg16 features
        self.net = VGG16(pretrained=True, requires_grad=False)
        self.conv0 = SingleConv(self.channels[0], 1, use_dropout=use_dropout)
        self.conv1 = SingleConv(self.channels[1], 1, use_dropout=use_dropout)
        self.conv2 = SingleConv(self.channels[2], 1, use_dropout=use_dropout)
        self.conv3 = SingleConv(self.channels[3], 1, use_dropout=use_dropout)
        self.conv4 = SingleConv(self.channels[4], 1, use_dropout=use_dropout)
        self.load_from_pretrained()

        # Freeze the network since this is metric not a trainable network.
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        # Download the pretrained VGG16 network.
        ckpt = get_ckpt_path(name='vgg_lpips', root=Path.home() / ".cache/iris/tokenizer_pretrained_vgg")
        self.load_state_dict(th.load(ckpt, map_location=th.device('cpu')), strict=False)

    def forward(self, sample: th.Tensor, target: th.Tensor) -> th.Tensor:
        input_sample, input_target = self.scaling_layer(sample), self.scaling_layer(target)
        out_sample, out_target = self.net(input_sample), self.net(input_target)

        feats_sample, feats_target, diffs = {}, {}, {}
        convs = [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4]
        for i in range(len(self.channels)):
            feats_sample[i], feats_target[i] = normalize_tensor(out_sample[i]), normalize_tensor(out_target[i])
            diffs[i] = (feats_sample[i] - feats_target[i]) ** 2     # squared difference

        res = [spatial_average(convs[i].model(diffs[i]), keepdim=True) for i in range(len(self.channels))]
        return reduce(lambda x, y: x + y, res)  # sum of all channels


def download(url: str, local_path: str, chunk_size: int = 1024):
    """
    Download a file from a given url to a local path.
    Args:
        url: The url to download the file from.
        local_path: The local path to save the file to.
        chunk_size: The size of the chunks to download the file in.
    """
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))    # in bytes
        with tqdm(total=total_size, unit='B', unit_scale=True) as prog_bar:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        prog_bar.update(chunk_size)


def md5_hash(path: str) -> str:
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    """
    Download a checkpoint from a given url to a local path.
    Args:
        name: The name of the checkpoint.
        root: The root directory to save the checkpoint to.
        check: If true, check the MD5 hash of the downloaded file.

    Returns:
        (str): The path to the downloaded checkpoint.
    """
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}...")
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], f"MD5 hash mismatch. Expected: {MD5_MAP[name]}. Got: {md5}."
    return path