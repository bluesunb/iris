import numpy as np
import torch as th
from torchvision.transforms import functional

from einops import rearrange
from PIL import Image
from typing import Tuple, Dict, Any

from src.agent import Agent
from src.envs import SingleProcessEnv, WorldModelEnv
from src.game.keymap import get_keymap_and_action_names


class AgentEnv:
    """
    Environment for the actor critic agent.
    """
    def __init__(self, agent: Agent, env: SingleProcessEnv, keymap_name: str, do_recon: bool = False):
        """
        Args:
            agent: actor-critic agent
            env: Single environment wrapped with SingleProcessEnv
            keymap_name: action keymap name
            do_recon: whether to use observation as reconstructed one
        """
        self.agent = agent
        self.env = env
        self.do_recon = do_recon
        self.action_names = get_keymap_and_action_names(keymap_name)[1]

        self.obs = None
        self.now = None
        self.gamma_return = None

    def _to_tensor(self, obs: np.ndarray) -> th.FloatTensor:
        """
        Convert observation (0 ~ 255) to tensor (0 ~ 1).
        Args:
            obs (np.ndarray): observation (0, 255) of shape (n H W C)
        Returns:
            (th.FloatTensor): observation (0, 1) of shape (n C H W)
        """
        assert isinstance(obs, np.ndarray) and obs.dtype == np.uint8, f"type: {type(obs)}, dtype: {obs.dtype}"
        return rearrange(th.FloatTensor(obs).div(255), 'n h w c -> n c h w').to(self.agent.device)

    def _to_array(self, obs: th.FloatTensor) -> np.ndarray:
        """
        Convert observation (0 ~ 1) to array (0 ~ 255).
        Args:
           obs (th.FloatTensor): observation (0, 1) of shape (n C H W)
        Returns:
            (np.ndarray): observation (0, 255) of shape (n H W C)
        """
        assert obs.ndim == 4 and obs.size(0) == 1, f"obs.shape: {obs.shape} != (1, C, H, W)"
        return obs[0].mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def step(self, *args, **kwargs) -> Tuple[th.Tensor, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Get action from the actor critic model, and step the environment.
        """
        with th.no_grad():
            action = self.agent.get_action_token(self.obs, sample=True)
            action = action.cpu().numpy()

        obs, reward, done, info = self.env.step(action)
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs
        self.now += 1
        self.gamma_return += reward[0]
        info = {'timestep': self.now,
                'action': self.action_names[action[0]],
                'return': self.gamma_return}

        return obs, reward, done, info

    def render(self) -> Image.Image:
        assert self.obs.size() == (1, 3, 64, 64), f"obs.shape: {self.obs.shape} != (1, 3, 64, 64)"
        if isinstance(self.env, SingleProcessEnv):
            original_obs = self.env.env.unwrapped.original_obs
        else:
            original_obs = self._to_array(self.obs)   # (1, H, W, C)

        if self.do_recon:
            recon = th.clamp(self.agent.tokenizer.encode_decode(self.obs, preprocess=True, postprocess=True), 0, 1)
            recon = functional.resize(recon,
                                      size=original_obs.shape[:2],
                                      interpolation=functional.InterpolationMode.NEAREST)
            resized_obs = functional.resize(self.obs,
                                            size=original_obs.shape[:2],
                                            interpolation=functional.InterpolationMode.NEAREST)

            recon = self._to_array(recon)
            resized_obs = self._to_array(resized_obs)
            img_arr = np.concatenate([original_obs, resized_obs, recon], axis=1)    # (H, 3 * W, C)
        else:
            img_arr = original_obs

        return Image.fromarray(img_arr)
