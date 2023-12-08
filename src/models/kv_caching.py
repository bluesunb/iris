import numpy as np
import torch as th
from typing import Any, Tuple, Optional


class AssignWithoutInplaceCheck(th.autograd.Function):
    """
    AssignWithoutInplaceCheck is a function that slice all dimension after `dim` from `start` to `stop` of x and inplace it to y.
    It may seem like a simple inplace operation, but it can auto-grad through `th.autograd`.

    Methods:
        forward(ctx, x, y, dim, start, stop) -> th.Tensor:
            Slice all dimension after `dim` from `start` to `stop` of x and inplace it to y.
        backward(ctx, grad_output) -> Tuple[Optional[th.Tensor], ...]]:
            Return the corresponding gradients of each input.
    """
    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        """
        `[(:::) , ... (dim) ..., (start:stop)]`
        => just slice all dimension after `dim` from start to stop.
        """
        return tuple([slice(None), ] * dim + [slice(start, stop), ])
    
    @staticmethod
    def forward(ctx: th.autograd.Function, x: th.Tensor, y: th.Tensor, dim: int, start: int, stop: int) -> th.Tensor:
        """
        Args:
            ctx: (th.autograd.Function): The context.
            x (th.Tensor): The tensor to assign to.
            y (th.Tensor): The tensor to assign from.
            dim (int): The dimension to assign to.
            start (int): The start index of the assignment.
            stop (int): The stop index of the assignment.

        Returns:
            (th.Tensor): The assigned tensor.
        """
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        # inplace x with y for all dimension after `dim` from start to stop
        x.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = y
        return x
    
    @staticmethod
    def backward(ctx: Any, grad_output: th.Tensor) -> Tuple[Optional[th.Tensor], ...]:
        """
        Args:
            ctx: (Any): The context.
            grad_output (th.Tensor): The gradient flow from the next layer.

        Returns:
            The corresponding gradients of each input.
            Note that we takes 5 inputs: x, y, dim, start, stop.
        """
        return (grad_output,    # grad_output for x
                grad_output[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], # grad_output for y
                None, None, None)   # grad_output for dim, start, stop -> no grad
    

class Cache:
    """
    Cache is a class that stores the keys or values of the attention layers.

    Attributes:
        _cache: (th.Tensor): The cache shape of (n_tokenss, n_heads, max_tokens, head_dim).
        _size: (int): The current size of the cache.
    """
    def __init__(self, n_tokens: int, n_heads: int, max_tokens: int, emb_dim: int, device: th.device):
        assert emb_dim % n_heads == 0, f"emb_dim({emb_dim}) must be divisible by n_heads({n_heads})"
        self._n_tokens = n_tokens
        self._n_heads = n_heads
        self._max_tokens = max_tokens
        self._emb_dim = emb_dim
        self._device = device
        self._cache = None
        self._size = None

        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        Returns:
            (Tuple[int, int, int, int]): n_tokens, n_heads, current size, head_dim
        """
        n_tokens, n_heads, max_tokens, head_dim = self._cache.shape
        return n_tokens, n_heads, self._size, head_dim

    def reset(self) -> None:
        head_dim = self._emb_dim // self._n_heads
        self._cache = th.empty(self._n_tokens, self._n_heads, self._max_tokens, head_dim, device=self._device)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        """
        Remove the some of the cache by the mask.

        Args:
            mask (np.ndarray): (n_tokenss,)
        """
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0], f"mask.shape({mask.shape}) must be {self.shape}"
        self._cache = self._cache[mask]     # (mask_true, n_heads, max_tokens, head_dim)
        self._n_tokens = self._cache.shape[0]

    def get(self) -> th.Tensor:
        """
        Get the current cache.

        Returns:
            (th.Tensor): (n_tokenss, n_heads, _size, head_dim)
        """
        return self._cache[:, :, :self._size, :]
    
    def update(self, x: th.Tensor) -> None:
        """
        Append x to the cache.
        To do this, x has same ndim and shape except for _size with the cache.
        Also, the key length of x must be smaller than remain space in the cache.
        """
        assert x.ndim == self._cache.ndim, f"x.ndim({x.ndim}) must be {self._cache.ndim}"
        # x and cache must have same shape except for _size
        assert all([x.size(i) == self._cache.size(i) for i in (0, 1, 3)]), f"x.shape({x.shape}) must be {self._cache.shape} for dim (0, 1, 3)"
        # key length of x must be smaller than remain space in the cache
        assert self._size + x.size(2) <= self._cache.shape[2], f"self._size({self._size}) + x.size(2)({x.size(2)}) must be <= self._cache.shape[2]({self._cache.shape[2]})"
        
        # store x to the cache
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 2, self._size, self._size + x.size(2))
        self._size += x.size(2)

    def __repr__(self):
        return f"Cache[{self.size}/{self._max_tokens}]"


class KVCache:
    """
    KVCache is a class that stores the keys and values cache of the attention layers.
    """
    def __init__(self, n_tokens: int, n_heads: int, max_tokens: int, emb_dim: int, device: th.device):
        self._k_cache = Cache(n_tokens, n_heads, max_tokens, emb_dim, device)
        self._v_cache = Cache(n_tokens, n_heads, max_tokens, emb_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """
        n_tokens, n_heads, _size, head_dim
        """
        return self._k_cache.shape
    
    def reset(self):
        self._k_cache.reset()
        self._v_cache.reset()
    
    def prune(self, mask: np.ndarray):
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[th.Tensor, th.Tensor]:
        return self._k_cache.get(), self._v_cache.get()
    
    def update(self, k: th.Tensor, v: th.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)

    def __repr__(self) -> str:
        return f"KVCache[{self._k_cache.size}/{self._k_cache._max_tokens}]"


class KeysValues:
    """
    Tuple of KVCache for each layer.
    """
    def __init__(self, 
                 n_tokens: int, 
                 n_heads: int, 
                 max_tokens: int, 
                 emb_dim: int, 
                 num_layers: int,
                 device: th.device):
        
        self._kv = tuple(KVCache(n_tokens, n_heads, max_tokens, emb_dim, device) for _ in range(num_layers))

    def __getitem__(self, key: int) -> KVCache:
        return self._kv[key]
    
    def __len__(self) -> int:
        return len(self._kv)
    
    @property
    def size(self) -> int:
        """
        return the current size of the cache.
        """
        return self._kv[0].shape[2]
    
    def reset(self):
        for kv_cache in self._kv:
            kv_cache.reset()

    def prune(self, mask: np.ndarray):
        for kv_cache in self._kv:
            kv_cache.prune(mask)

    def __repr__(self) -> str:
        return str(self._kv[:2]) + f"...({len(self._kv)})"
