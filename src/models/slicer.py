import math
import torch as th
import torch.nn as nn
from typing import List, Union


class Slicer(nn.Module):
    """
    Slice the input tensor to keep according to the given block mask.
    Attributes:
        block_size: (int): number of tokens in a block (tok_obs + tok_act)
        n_kept_tokens (int): number of tokens to keep
                             => Note that block_mask: (block_size, ), sum(block_mask) = n_kept_tokens
        indices (th.Tensor): (max_blocks * n_kept_tokens, ),
            indices of kept tokens when we flatten the blocks
            = [0, 2, 4, 5] * max_block +
            [0, 0, 0, 0] + block_size * [1, 1, 1, 1] + ... + block_size * [max_blocks - 1, ...]
    """
    def __init__(self, max_blocks: int, block_mask: th.Tensor):
        """
        Args:
            max_blocks (int): The maximum number of blocks to sample.
            block_mask (th.Tensor): (n_tokens, ), True if token is kept, False otherwise.
        """
        super().__init__()
        self.block_size = block_mask.size(0)
        self.n_kept_tokens = block_mask.sum().long().item()     # count true values
        
        kept_indices = th.where(block_mask)[0].repeat(max_blocks)   # (n_kept_tokens * max_blocks, )
        # (1, ...(n_kept_tokens)..., 1, 2, 2...., )
        offsets = th.arange(max_blocks).repeat_interleave(self.n_kept_tokens)   # (n_kept_tokens * max_blocks, )

        # kept_indices = [0, 2, 4, 5] * max_blocks -> we kept 4 tokens for all valid blocks
        # offsets = [0, 0, 0, 0] + [1, 1, 1, 1] + ... + [max_blocks - 1, ...]
        # kept_indices + self.block_size * offsets -> each index to kept when we flatten the blocks
        self.register_buffer("indices", kept_indices + self.block_size * offsets)   # (max_blocks * n_kept_tokens, )

    def compute_slice(self, n_steps: int, prev_steps: int = 0) -> th.Tensor:
        """
        Select indices to keep for the given number of steps.
        Args:
            n_steps (int): The number of steps to compute.
            prev_steps (int): The number of steps already computed.
        """
        total_steps = n_steps + prev_steps
        n_blocks = math.ceil(total_steps / self.block_size)
        indices: th.Tensor = self.indices[:n_blocks * self.n_kept_tokens]
        # select indices in [prev_steps, total_steps)
        indices = indices[th.logical_and(prev_steps <= indices, indices < total_steps)]
        return indices - prev_steps    # relative position in the block
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Slicer is not a layer, it should not be called directly")


class Head(Slicer):
    """
    Slice the input tensor to keep only the tokens for the given number of steps.
    Then, apply the head module to the sliced tensor.
    """
    def __init__(self, max_blocks: int, block_masks: th.Tensor, head_module: nn.Module):
        super().__init__(max_blocks, block_masks)
        self.head_module = head_module

    def forward(self, x: th.Tensor, n_steps: int, prev_steps: int) -> th.Tensor:
        x_sliced = x[:, self.compute_slice(n_steps, prev_steps)]
        return self.head_module(x_sliced)


class Embedder(nn.Module):
    """
    Embed tokens which is masked to keep with each embedding table.

    for tokens (bs, t, n)

    
    Attributes:
        embed_tables (List[nn.Embedding]): (n_masks, ) list of embedding tables
        slicers (List[Slicer]): (n_masks, ) slicers for each makse
    """
    def __init__(self,
                 max_blocks: int,
                 block_masks: List[th.Tensor],
                 embed_tables: Union["nn.ModuleList[nn.Embedding]", List[nn.Embedding]]):
        """
        Args:
            max_blocks (int): The maximum number of blocks to sample.
            block_masks (List[th.Tensor]): (n_masks, n_tokens), True if token is kept, False otherwise.
            embed_tables (List[nn.Embedding]): (n_masks, ) list of embedding tables
        """
        super().__init__()
        assert len(block_masks) == len(embed_tables), \
            f"len(block_masks):{len(block_masks)} != len(embed_tables):{len(embed_tables)}"
        assert (sum(block_masks) == 1).all(), "block_masks should be mutually exclusive"

        self.embed_tables = embed_tables
        self.emb_dim = embed_tables[0].embedding_dim
        assert all([e.embedding_dim == self.emb_dim for e in embed_tables]), \
            f"All embedding tables should embed the given tensor to dimension with {self.emb_dim}"

        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: th.Tensor, n_steps: int, prev_steps: int) -> th.Tensor:
        """
        Args:
            tokens (th.Tensor): (bs, t, n), where n is the number of tokens.
                                Note that token is assigned to each pixel in the observation.
            n_steps: number of steps (s <= t) to compute embeddings for each token.
            prev_steps: number of steps already computed.

        Returns:
            (th.Tensor): (bs, t, e), where e is the embedding dimension.
            We give the tokens (bs, t, n) where n is the number of tokens.
            n must be equal to the size of flatten obs (n = hw).
            Then, each slicer will select the indices to keep in the flatten obs by block_size.
        """
        assert tokens.ndim == 2, f"tokens should have 2 dimensions, got {tokens.ndim}"  # (bs, t)
        outputs = th.zeros(*tokens.shape, self.emb_dim, device=tokens.device)   # (bs, t, e), store embeddings of each token
        for slicer, emb in zip(self.slicers, self.embed_tables):
            s = slicer.compute_slice(n_steps, prev_steps)   # valid maximum `n_step` number of indices where we should keep the token
            outputs[:, s] = emb(tokens[:, s])   # select the tokens for given block+mask and compute embeddings
        return outputs


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    input_tensor = th.randint(0, vocab_size, (batch_size, seq_len))

    emb_dim = 7
    emb_tables = [nn.Embedding(vocab_size, emb_dim) for _ in range(seq_len)]
    block_masks = list(th.eye(seq_len).repeat_interleave(3, dim=1))

    max_blocks = 2
    embedder = Embedder(max_blocks, block_masks, emb_tables)

    n_steps = seq_len
    prev_steps = 0
    output = embedder(input_tensor, n_steps, prev_steps)
    print(output.shape)