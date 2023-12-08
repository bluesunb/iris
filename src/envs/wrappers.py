import gym
import numpy as np
from PIL import Image
from gym import spaces
from gym.envs.atari import AtariEnv

from typing import Tuple


def make_atari(env_id: str,
               size: int = 64,
               max_episode_steps: int = None,
               noop_max: int = 30,
               frame_skip: int = 4,
               done_on_life_loss: bool = False,
               clip_reward: bool = False):
    """
    Create an Atari environment with some standard preprocessing.

    Args:
        env_id: The Atari game ID.
        size: The size of the observation after preprocessing.
        max_episode_steps: The maximum number of steps per episode.
        noop_max: The maximum number of no-op steps at the beginning of each episode.
        frame_skip: The number of frames to skip per action.
        done_on_life_loss: Whether to consider a life loss as done.
        clip_reward: Whether to clip the reward to [-1, 1].
    """

    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id or "Deterministic" in env.spec.id, \
        "Please use the NoFrameskip or Deterministic version of the Atari environment."

    env = ResizeObsWrapper(env, size=(size, size))
    if clip_reward:
        env = RewardClipWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    """
    Resize observation space to the given size.

    Attributes:
        size (Tuple[int, int]): The size of the observation after resizing.
        unwrapped.original_obs (np.ndarray): The original observation before resizing.
    """

    def __init__(self, env: AtariEnv, size: Tuple[int, int]):
        """
        Args:
            env (AtariEnv): The Atari environment to resize its observation space.
            size (Tuple[int, int]): The size of the observation after resizing.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        # image space
        self.observation_space = spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray) -> np.ndarray:
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """This is called in the step() method of the Atari environment."""
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClipWrapper(gym.RewardWrapper):
    """
    Return the sign of the reward.
    """

    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0 (do nothing action).

    Attributes:
        noop_max (int): The maximum number of no-op steps at the beginning of each episode.
        num_noop_steps (int): The number of no-op steps at the beginning of the current episode.
        noop_action (int): The assigned integer for no-op action.
    """

    def __init__(self, env: AtariEnv, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.num_noop_steps = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP', "Please make sure that the first action is 'NOOP'."

    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.num_noop_steps is not None:
            num_noop_steps = self.num_noop_steps
        else:
            num_noop_steps = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert num_noop_steps > 0, "No-op steps must be greater than 0."
        obs = None
        for _ in range(num_noop_steps):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make (end-of-life == end-of-episode), but only reset for true game over.
    ==> In the other word, life done means episode done but not a reset condition.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """

    def __init__(self, env: AtariEnv):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done

        lives: int = self.env.unwrapped.ale.lives()
        done = 0 < lives < self.lives
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, reward, done, info = self.env.step(0)  # Do no-op.

        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame.

    Attributes:
        _obs_buffer (np.ndarray): The buffer for the last two observations. (two obs = transition)
        _skip (int): The number of frames to skip per action.
        max_frame (np.ndarray): The maximum frame of the last two observations.
    """

    def __init__(self, env: AtariEnv, skip=4):
        gym.Wrapper.__init__(self, env)
        assert skip > 0, "Skip parameter must be greater than 0."
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)  # (2, h ,w, c)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)  # (h, w, c)

    def step(self, action: int):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:  # second last frame
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:  # last frame
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        self.max_frame = self._obs_buffer.max(axis=0)
        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
