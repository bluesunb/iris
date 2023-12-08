import numpy as np
import torch as th
from einops import rearrange
from pathlib import Path
from PIL import Image

from src.dataset import Batch
from src.models.tokenizer import Tokenizer

BATCH_KEYS = ['actions', 'ends', 'mask_padding', 'observation', 'rewards']


def check_batch(batch: Batch):
    """
    Check if the batch has correct keys and shapes.
    """
    assert sorted(batch.keys()) == BATCH_KEYS, f'batch.keys() ({sorted(batch.keys())}) != BATCH_KEYS ({BATCH_KEYS})'
    bs, t, c, h, w = batch['observation'].shape
    assert batch['actions'].shape == (bs, t), f'batch actions shape error: {batch["actions"].shape} != {(bs, t)}'
    assert batch['rewards'].shape == (bs, t), f'batch rewards shape error: {batch["rewards"].shape} != {(bs, t)}'
    assert batch['ends'].shape == (bs, t), f'batch ends shape error: {batch["ends"].shape} != {(bs, t)}'
    assert batch['mask_padding'].shape == (bs, t), \
        f'batch mask_padding shape error: {batch["mask_padding"].shape} != {(bs, t)}'


def check_float_btw_0_1(inputs: th.Tensor):
    """
    Check if the tensor is floating point and between 0 and 1.
    """
    assert inputs.is_floating_point(), f'inputs ({inputs}) is not floating point'
    assert inputs.min() >= 0 and inputs.max() <= 1, f'inputs ({inputs}) is not between 0 and 1'


def tensor_to_np_frames(inputs: th.Tensor) -> np.ndarray:
    """
    Convert the tensor to numpy array of image frames. (0 ~ 255)
    """
    check_float_btw_0_1(inputs)
    return inputs.mul(255).cpu().numpy().astype(np.uint8)


@th.no_grad()
def reconstruct_by_tokenizer(inputs: th.Tensor, tokenizer: Tokenizer) -> th.Tensor:
    """
    Reconstruct the inputs by the tokenizer.
    """
    check_float_btw_0_1(inputs)
    reconstructions = tokenizer.encode_decode(inputs, preprocess=True, postprocess=True)
    return th.clamp(reconstructions, 0, 1)


@th.no_grad()
def generate_recon_with_tokenizer(batch: Batch, tokenizer: Tokenizer):
    """
    Generate the reconstruction frames from the batch images through the tokenizer.
    """
    check_batch(batch)
    inputs = rearrange(batch['observation'], 'b t c h w -> (b t) c h w')  # image-wise flatten
    outputs = reconstruct_by_tokenizer(inputs, tokenizer)  # (bt, c, h, w)
    b, t, c, h, w = batch['observation'].shape
    outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    recon_frames = tensor_to_np_frames(outputs)
    return recon_frames


@th.no_grad()
def make_recon_from_batch(batch: Batch, save_dir: Path, epoch: int, tokenizer: Tokenizer):
    check_batch(batch)
    original_frames = tensor_to_np_frames(
        rearrange(batch['observation'], 'b t c h w -> b t h w c'))
    all_frames = [original_frames]

    recon_frames = generate_recon_with_tokenizer(batch, tokenizer)
    all_frames.append(recon_frames)

    tmp_frames = np.concatenate([original_frames, recon_frames], axis=-2)  # (b, t, 2h, w, c)
    tmp_frames = np.concatenate(list(tmp_frames), axis=-3)  # (b, 2t, 2h, w, c)

    for i, image in enumerate(map(Image.fromarray, tmp_frames)):
        image.save(save_dir / f'epoch_{epoch:03d}_t_{i:03d}.png')
