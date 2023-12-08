import gym
import numpy as np
from typing import Any, Tuple, Callable, Dict, Any
from src.envs.done_tracker import DoneTrackerEnv


class SingleProcessEnv(DoneTrackerEnv):
    """
    Single process environment, so we have only one environment.
    """
    def __init__(self, env_fn: Callable[[], gym.Env]):
        """
        Args:
            env_fn: Function that returns a gym.Env instance.
        """
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space.n
    
    def should_reset(self) -> bool:
        return self.num_envs_done > 0
    
    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        obs = self.env.reset()  # obs: np.ndarray
        return obs[None, ...]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Single process environment step, so batched return has batch size 1.
        Args:
            action (np.ndarray): Action for each environment. In this case, batch size is 1. (1, 1)
        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray, Any]): observation, reward, done, info

            In this case, batch size is 1 so, each shape is following:
                - observation: (1, C, H, W)
                - reward: (1)
                - done: (1)
                - info: Any
        """
        obs, reward, done, info = self.env.step(action[0])
        done = np.array([done], dtype=np.uint8)
        self.update_done_tracker(done)
        return obs[None, ...], np.array([reward]), done, info
    
    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
