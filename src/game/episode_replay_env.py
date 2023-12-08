import numpy as np
import torch as th

from pathlib import Path
from PIL import Image
from einops import rearrange

from src.episode import Episode, EPISODE_MODES
from src.game.keymap import get_keymap_and_action_names, EPISODE_REPLAY_ACTION_NAMES

from typing import List, Tuple, Dict, Union, Optional, Any


class EpisodeReplayEnv:
    def __init__(self, replay_keymap_name: str, episode_dir: Path):
        assert episode_dir.is_dir(), f'Episode directory {episode_dir} does not exist.'
        keymap, action_names = get_keymap_and_action_names(replay_keymap_name)
        self.action_names = action_names
        self._paths = {}
        for mode in EPISODE_MODES:
            directory = episode_dir / mode
            if directory.is_dir():
                # episode_*.pt
                self._paths[mode] = sorted([p for p in directory.iterdir() if 'episode_' in p.stem and p.suffix == '.pt'])
                print(f"Found {len(self._paths[mode])} {mode} episodes in {directory}")
            else:
                print(f"There are no {mode} episodes in {directory}")

        self.now = None
        self.episode: Optional[Episode] = None  # current episode
        self.episode_idx = 0
        self.mode = 'train'
        self.load()

    @property
    def paths(self) -> List[Path]:
        """Name of episodes path for the current mode."""
        return self._paths[self.mode]

    @property
    def observations(self):
        return self.episode.observations

    @property
    def actions(self):
        return self.episode.actions

    @property
    def rewards(self):
        return self.episode.rewards

    @property
    def ends(self):
        return self.episode.ends

    def load(self) -> None:
        """Load the episode at the current index from the paths"""
        self.episode = Episode(**th.load(self.paths[self.episode_idx]))
        self.now = 0

    def load_next(self) -> None:
        """Load the next episode from the paths"""
        self.episode_idx = (self.episode_idx + 1) % len(self.paths)
        self.load()

    def load_previous(self) -> None:
        """Load the previous episode from the paths"""
        self.episode_idx = (self.episode_idx - 1) % len(self.paths)
        self.load()

    def set_mode(self, mode: str):
        assert mode in EPISODE_MODES, f'mode({mode}) must be one of {EPISODE_MODES}'
        if mode in self._paths:
            self.mode = mode
            self.episode_idx = 0
            self.load()
        else:
            print(f"There are no {mode} episodes")

    def reset(self):
        return self.observations[self.now]

    def step(self, action: int) -> Tuple[th.Tensor, float, bool, Dict[str, Any]]:
        """
        Refer the `src.game.keymap.EPISODE_REPLAY_ACTION_NAMES` for each action idx.
        Args:
            action (int): numeric action index that one of EPISODE_REPLAY_ACTION_NAMES
        Returns:
            observation, reward, done, info
        """
        if action == 1:     # previous
            self.now = (self.now - 1) % len(self.episode)
        elif action == 2:   # next
            self.now = (self.now + 1) % len(self.episode)
        if action == 3:     # previous_10
            self.now = (self.now - 10) % len(self.episode)
        elif action == 4:   # next_10
            self.now = (self.now + 10) % len(self.episode)
        elif action == 5:   # go_to_start
            self.now = 0
        elif action == 6:   # load_previous
            self.load_previous()
        elif action == 7:   # load_next
            self.load_next()
        elif action == 8:   # go_to_train_episodes
            self.set_mode('train')
        elif action == 9:   # go_to_test_episodes
            self.set_mode('test')
        elif action == 10:  # go_to_imagination_episodes
            self.set_mode('imagination')

        action = self.actions[self.now]
        reward = self.rewards[self.now].item()
        done = self.ends[self.now].item()
        info = {'episode_name': f"[{self.mode} {self.paths[self.episode_idx].stem}",
                'timestep': self.now,
                'action': self.action_names[action],
                'cum_reward': f'{sum(self.rewards[:self.now + 1]):.3f}'}

        return self.observations[self.now], reward, done, info

    def render(self) -> Image.Image:
        obs = self.observations[self.now]     # [C, H, W]
        arr = obs.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(arr)

    def __len__(self) -> int:
        """Length of the episode."""
        return len(self.ends)