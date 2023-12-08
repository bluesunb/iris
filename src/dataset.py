import math, random
import psutil
import torch as th

from functools import partial
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Deque, Set

from .episode import Episode

"""
Batch:
    observations: th.ByteTensor
    actions: th.LongTensor
    rewards: th.FloatTensor
    ends: th.LongTensor
    mask_padding: th.BoolTensor
"""
Batch = Dict[str, th.Tensor]


class EpisodeDataset:
    """
        A dataset that stores episodes.

        Attributes:
            max_num_episodes (int):                 The maximum number of episodes to store in the dataset.
            name (str):                             The name of the dataset.
            num_seen_episodes (int):                The number of episodes seen by the dataset. Unlike the length of the dataset, this number is not affected by deletion (popleft).
            episodes (Deque[Episode]):              A deque of episodes.
            episode_id2queue_idx (Dict[int, int]):  A mapping from episode id to queue index.
            modified_episodes (Set[int]):           A set of episode ids that have been modified.
            deleted_episodes (Set[int]):            A set of episode ids that have been deleted.
    """
    def __init__(self, 
                 max_num_episodes: Optional[int] = None, 
                 name: Optional[str] = None):
        """
        Args:
            max_num_episodes (int):     The maximum number of episodes to store in the dataset.
            name (str):                 The name of the dataset.
        """
        
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else "dataset"
        self.num_seen_episodes = 0
        self.episodes: Deque[Episode] = deque()
        self.episode_id2queue_idx: Dict[int, int] = dict()
        self.modified_episodes: Set[int] = set()
        self.deleted_episodes: Set[int] = set()

    def __len__(self):
        return len(self.episodes)
    
    def clear(self) -> None:
        self.episodes.clear()
        self.episode_id2queue_idx.clear()

    def add_episode(self, episode: Episode) -> int:
        """
        Add an episode to the dataset. 
        If the dataset is full, remove the oldest episode.
        mapping from episode id to queue index and set of modified episodes idx and deleted episodes idx are update accordingly.

        Args:
            episode (Episode): The episode to add.
            
        Returns:
            (int): The episode id of the added episode.
        """
        if self.max_num_episodes is not None and len(self.episodes) == self.max_num_episodes:
            self._popleft()    # remove the oldest episode from the dataset
        episode_id = self._append_episode(episode)
        return episode_id
    
    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.episode_id2queue_idx, f"Episode with id {episode_id} not found."
        queue_idx = self.episode_id2queue_idx[episode_id]
        return self.episodes[queue_idx]
    
    def update_episode(self, episode_id: int, episode: Episode) -> None:
        """
        Update an episode of episode_id by merging new episode into it.
        """
        assert episode_id in self.episode_id2queue_idx, f"Episode with id {episode_id} not found."
        queue_idx = self.episode_id2queue_idx[episode_id]
        merged_episode = self.episodes[queue_idx].merge(episode)
        self.episodes[queue_idx] = merged_episode
        self.modified_episodes.add(episode_id)

    def _popleft(self) -> Episode:
        """
        Remove the oldest episode from the dataset.
        """
        delete_id = [k for k, v in self.episode_id2queue_idx.items() if v == 0]
        assert len(delete_id) == 1, "There should be exactly one episode with queue index 0."
        self.deleted_episodes.add(delete_id[0])
        self.episode_id2queue_idx = {k: v - 1 for k, v in self.episode_id2queue_idx.items() if v > 0}
        return self.episodes.popleft()
    
    def _append_episode(self, episode: Episode) -> int:
        """
        - [1] Append the episode to the dataset.
        - [2] Update the number of seen episodes and the mapping from episode id to queue index.
        - [3] Add the episode id to the set of modified episodes.

        Args:
            episode (Episode): The episode to append.

        Returns:
            (int): The episode id of the appended episode.
        """
        episode_id = self.num_seen_episodes
        self.episode_id2queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        self.num_seen_episodes += 1
        self.modified_episodes.add(episode_id)
        return episode_id
    
    def sample_batch(self, batch_num_samples:int, seq_len: int, sample_from_start: bool = True) -> Batch:
        """
        Randomly extract segments from the episodes and stack them into a single dict with batch of tensors.

        Args:
            batch_num_samples (int): The number of episodes segments to sample.
            seq_len (int): The length of each episode segment.
            sample_from_start (bool): If true, there is no left padding. Otherwise, there is no right padding.

        Returns:
            (Batch): A batch of dict that contains collected episodes segments.
        """
        return self._collate_episodes_segments(
            self._sample_episodes_segments(batch_num_samples, seq_len, sample_from_start)
        )

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        """
        Collect all segments of the same key and stack them into a batch.

        Args:
            episodes_segments (List[Episode]): A list of episodes segments.

        Returns:
            (Batch): A batch of dict that contains collected episodes segments.
        """
        ep_dict_segments = [ep_seg.__dict__ for ep_seg in episodes_segments]
        batch = dict()
        for k in ep_dict_segments[0]:
            # collect all segments of the same key
            batch[k] = th.stack([ep_seg[k] for ep_seg in ep_dict_segments])
        batch['observations'] = batch['observations'].float() / 255.0   # int8 -> float & scale to [0, 1]
        return batch
        
    def _sample_episodes_segments(self, batch_num_samples: int, seq_len: int, sample_from_start: bool = True) -> List[Episode]:
        """
        Sample a batch of episodes segments from the dataset.

        Args:
            batch_num_samples (int): The number of episodes segments to sample.
            seq_len (int): The length of each episode segment.
            sample_from_start (bool): If true, there is no left padding. Otherwise, there is no right padding.

        Returns:
            (Batch): A batch of episodes segments.
        """
        sampled_episodes = random.choices(self.episodes, k=batch_num_samples)
        sampled_episodes_segments: List[Episode] = []
        for sampled_ep in sampled_episodes:
            if sample_from_start:
                start = random.randint(0, batch_num_samples - 1)
                stop = start + seq_len
            else:
                stop = random.randint(1, batch_num_samples)
                start = stop - seq_len

            sampled_episodes_segments.append(sampled_ep.segment(start, stop, should_pad=True))
            assert len(sampled_episodes_segments[-1]) == seq_len, "Sampled episode segment has incorrect length."
        
        return sampled_episodes_segments
    
    def traverse(self, batch_num_samples: int, chunk_size: int) -> Batch:
        """
        Traverse the dataset and yield a batch of episodes segments.
        """
        for episode in self.episodes:
            chuck_len = math.ceil(len(episode) / chunk_size)     # episodes are divided into chunk size
            chunk = [episode.segment(start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True) for i in range(chuck_len)]
            batch_len = math.ceil(len(chunk) / batch_num_samples)     # chunk are divided into batch num samples
            batches = [chunk[i * batch_num_samples : (i + 1) * batch_num_samples] for i in range(batch_len)]
            for batch in batches:
                yield self._collate_episodes_segments(batch)

    def update_disk_checkpoint(self, directory: Path) -> None:
        """
        Save modified episodes and remove deleted episodes from disk.
        The name of saved file is the episode id.
        """
        assert directory.is_dir(), f"Directory {directory} does not exist."
        for episode_id in self.modified_episodes:
            self.get_episode(episode_id).save(directory / f'{episode_id}.pt')
        for episode_id in self.deleted_episodes:
            (directory / f'{episode_id}.pt').unlink()
        
        self.modified_episodes.clear()
        self.deleted_episodes.clear()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir(), f"Directory {directory} does not exist."
        assert len(self.episodes) == 0, "Dataset should be empty before loading checkpoint."
        episode_ids = sorted([int(f.stem) for f in directory.iterdir()])    # all episode ids under the directory
        self.num_seen_episodes = episode_ids[-1] + 1    # the number of loaded episodes
        for episode_id in episode_ids:
            episode = Episode(**th.load(directory / f'{episode_id}.pt'))
            self.episode_id2queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)


class EpisodeRamMonitoringDataset(EpisodeDataset):
    """
    A Episode dataset that monitors ram usage by considering the number of steps. (not the number of episodes)
    If the ram usage exceeds the maximum usage, remove the oldest episode from the dataset.

    Attributes:
        max_ram_usage (str): The maximum ram usage.
        num_steps (int): The total number of steps in the episodes. This attribute is used to monitor ram usage.
        max_num_steps (int): The maximum number of steps in the episodes when ram usage exceeds the maximum usage.
    """
    def __init__(self, max_ram_usage: str, name: Optional[str] = None):
        """
        Args:
            max_ram_usage (str): The maximum ram usage. It should be a one of the following:
                1. A percentage string (e.g. '80%')
                2. A string with 'G' (e.g. '10G')
            name (str): The name of the dataset.
        """
        super().__init__(max_num_episodes=None, name=name)
        assert isinstance(max_ram_usage, str), "max_ram_usage should be a string."
        self.max_ram_usage = max_ram_usage
        self.num_steps = 0
        self.max_num_steps = None
        
    def check_ram_usage(self) -> bool:
        """
        Check if the ram usage exceeds the maximum usage.
        """
        if self.max_ram_usage.endswith('%'):
            max_usage = int(self.max_ram_usage.split('%')[0])
            assert 0 < max_usage < 100, "max_ram_usage should be a percentage. (0 ~ 100)"
            return psutil.virtual_memory().percent > max_usage
        
        elif self.max_ram_usage.endswith('G'):
            max_usage = float(self.max_ram_usage.split('G')[0])
            return psutil.Process().memory_info()[0] / 2.0 ** 30 > max_usage
        
        else:
            raise ValueError("max_ram_usage should end with '%' or 'G'.")
        
    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode: Episode) -> int:
        """
        Set the maximum number of steps when ram usage exceeds the maximum usage then, add the episode to the dataset.

        Returns:
            (int): The episode id of the added episode.
        """
        if self.max_num_episodes is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps     # record the number of steps when ram usage exceeds the maximum usage
        
        self.num_steps += len(episode)
        while self.max_num_steps is not None and self.num_steps > self.max_num_steps:
            self._popleft()     # if ram usage exceeds, remove the oldest episode from the dataset
        episode_id = self._append_episode(episode)
        return episode_id
    
    def _popleft(self) -> Episode:
        episode = super()._popleft()
        self.num_steps -= len(episode)
        return episode
