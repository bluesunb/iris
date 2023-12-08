import random
import gym
import numpy as np
import torch as th
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torchvision.transforms.functional import to_tensor, resize
from einops import rearrange

from PIL import Image
from typing import List, Optional, Tuple, Union

from src.models.kv_caching import KeysValues
from src.models.tokenizer import Tokenizer
from src.models.world_model import WorldModel


class WorldModelEnv:
    """
    Attributes:
        world_model_kvs (KeyValues): key-value cache of the world model
        obs_tokens (th.LongTensor): (b, k=hw), observation tokens
        _num_obs_tokens (int): number of observation tokens (k)
    """
    def __init__(self, 
                 tokenizer: Tokenizer,
                 world_model: WorldModel,
                 device: Union[str, th.device] = "cuda",
                 env: Optional[gym.Env] = None):
        
        self.device = device
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.env = env

        self.world_model_kvs: KeysValues = None
        self.obs_tokens: th.LongTensor = None
        self._num_obs_tokens: int = None

    @property
    def num_obs_tokens(self) -> int:
        return self._num_obs_tokens
    
    @th.no_grad()
    def reset(self) -> th.FloatTensor:
        """
        Generate initial observation, 
        then resets the environment and the key-value cache from the initial observation.

        Returns:
            List[PIL.Image]: Reconstructed images of the initial observation. 
                             It may be different from the actual initial observation.
        """
        assert self.env is not None, "No environment provided."
        # get initial observation from the environment
        obs = to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W)
        obs = resize(obs, (64, 64))  # (1, C, 64, 64)
        return self.reset_from_initial_obs(obs)     # List[(1, C, H, W)]
    
    @th.no_grad()
    def reset_from_initial_obs(self, initial_obs: th.FloatTensor) -> th.FloatTensor:
        """
        Resets the observation tokens and the key-value cache from the initial observation.

        Args:
            initial_obs (th.FloatTensor): (b, c, H, W) initial observation

        Returns:
            List[PIL.Image]: (b, c_out, H, W) Reconstructed images of the given initial observation.
        """
        initial_obs_tokens = self.tokenizer.encode(initial_obs, preprocess=True).tokens     # (b, c, H, W) -> (b, hw)
        b, num_obs_tokens = initial_obs_tokens.shape
        if self.num_obs_tokens is None:
            self._num_obs_tokens = num_obs_tokens   # hw
        
        self.refresh_kvs(initial_obs_tokens)
        self.obs_tokens = initial_obs_tokens
        return self.decode_obs_tokens()
    
    @th.no_grad()
    def step(self,
             action: Union[int, np.ndarray, th.LongTensor],
             pred_next_obs: bool = True) -> Tuple[Optional[th.FloatTensor], float, bool, dict]:
        """
        From the given action, generates the next observation, reward, and done flag.

        Args:
            action: action to take
            pred_next_obs (bool, optional): whether to predict the all tokens for next observation. Defaults to True.

        Returns:
            Tuple[th.FloatTensor, float, bool, dict]: next observation, reward, done flag, and info

                - next observation: (b, c_out, H, W)
                - reward: (b)
                - done: (b)
        """
        assert self.world_model_kvs is not None, "Key-value cache not initialized. Do reset() first."
        assert self.num_obs_tokens is not None, "Observation tokens not initialized. Do reset() first."
        
        # length of sequence to be generated. :=> tok_(t+1) ~ G(tok_t, a_t)
        # ex) ([r_1, d_1], x_21, x_22, ... , x_2k) => seq_len = 1 + k
        seq_len = 1 + self.num_obs_tokens if pred_next_obs else 1
        output_sequences, obs_tokens = [], []

        # if exceeds max tokens, refresh the key-value cache
        if self.world_model_kvs.size + seq_len > self.world_model.config.max_tokens:
            self.refresh_kvs(self.obs_tokens)
        
        token = action.clone().detach() if isinstance(action, th.Tensor) else th.tensor(action, dtype=th.long)
        token = token.reshape(-1, 1).to(self.device)  # (b, 1)

        for i in range(seq_len):
            world_model_output = self.world_model.forward(token, prev_kvs=self.world_model_kvs)
            output_sequences.append(world_model_output.output_sequence)     # (b, 1, e)
            
            if i == 0:  # first prediction is reward and done about the previous action
                reward = Categorical(logits=world_model_output.logits_rewards).sample()  # (b, 1)
                done = Categorical(logits=world_model_output.logits_ends).sample()  # (b, 1)
                
                reward = reward.float().cpu().numpy().reshape(-1)   # (b,)
                done = done.cpu().numpy().astype(bool).reshape(-1)  # (b,)
            
            if i < self.num_obs_tokens: # next observation tokens prediction
                obs_token = Categorical(logits=world_model_output.logits_observations).sample()     # (b, 1)
                obs_tokens.append(obs_token)
        
        output_sequences = th.cat(output_sequences, dim=1)  # (b, seq_len, e)
        self.obs_tokens = th.cat(obs_tokens, dim=1)         # (b, seq_len - 1 = hw)

        next_obs = self.decode_obs_tokens() if pred_next_obs else None  # (b, c_out, H, W)
        return next_obs, reward, done, {}

    @th.no_grad()
    def refresh_kvs(self, initial_obs_tokens: th.LongTensor) -> th.FloatTensor:
        """
        Refreshes the key-value cache with initial observations.
        To do this, we first generte empty key-value paris, then we pass the observations through the world model.

        Args:
            obs_tokens (th.LongTensor): (b, k=hw), initial observations
        
        Returns:
            th.FloatTensor: (b, k, e), transformer output sequence from the world model
        """
        b, num_obs_tokens = initial_obs_tokens.shape
        assert num_obs_tokens == self.num_obs_tokens, \
            f"num_obs_tokens:{num_obs_tokens} != self.num_obs_tokens:{self.num_obs_tokens}"
        self.world_model_kvs = self.world_model.transformer.generate_empty_kvs(n_keys=b, 
                                                                               max_tokens=self.world_model.config.max_tokens)
        world_model_outputs = self.world_model.forward(initial_obs_tokens, prev_kvs=self.world_model_kvs)
        return world_model_outputs.output_sequence  # (b, k, e)
    
    @th.no_grad()
    def decode_obs_tokens(self) -> th.Tensor:
        """
        Decodes the observation tokens into a list of PIL images.

        Returns:
            (List[PIL.Image]): list of reconstructed images (b, c_out, H, W)
        """
        tokens_embed = self.tokenizer.embedding(self.obs_tokens)  # (b, hw, e))
        z = rearrange(tokens_embed, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_obs_tokens)))
        reconstructed = self.tokenizer.decode(z, postprocess=True)  # (b, c_out, H, W)
        return th.clamp(reconstructed, 0, 1)

    @th.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c')
        frames = frames.mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]
    
    @th.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_obs_tokens), \
            f"obs_tokens.shape:{self.obs_tokens.shape} != (1, {self.num_obs_tokens})"
        self.render_batch()[0]


if __name__ == "__main__":
    from src.models.tokenizer import AutoencoderConfig, Encoder, Decoder
    from src.models.world_model import TransformerConfig
    # breackout env

    ae_config = AutoencoderConfig(resolution=64,
                                  in_channels=3,
                                  z_channels=128,
                                  out_channels=24,
                                  channel=32,
                                  channel_scales=(2, 3, 4),
                                  num_res_blocks=2,
                                  attn_resolutions=(64, 128,),
                                  dropout=0.0)
    
    tf_config = TransformerConfig(tokens_per_block=16*16+1,
                                  max_blocks=5,
                                  attention='block_casual',
                                  n_layers=2,
                                  n_heads=4,
                                  emb_dim=128,
                                  emb_dropout=0.0,
                                  residual_dropout=0.0,
                                  attn_dropout=0.0)
    
    obs_vocab_size = 10
    act_vocab_size = 19

    encoder = Encoder(ae_config)
    decoder = Decoder(ae_config)
    tokenizer = Tokenizer(vocab_size=obs_vocab_size, emb_dim=144, encoder=encoder, decoder=decoder, lpips=False)

    world_model = WorldModel(obs_vocab_size, act_vocab_size, config=tf_config)

    bs = 3
    c, h, w = 3, 64, 64

    import gym
    world_model_env = WorldModelEnv(tokenizer, world_model, env=gym.make("Breakout-v4"))
    obs = world_model_env.reset()
    world_model_env.step(1.0, pred_next_obs=False)