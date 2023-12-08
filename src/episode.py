import torch as th
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path


EPISODE_MODES = ['train', 'test', 'imagination']


@dataclass
class EpisodeMetrics:
    episode_length: int
    episode_return: float


@dataclass
class Episode:
    """
    An episode is a sequence of observations, actions, rewards and episode ends.

    Args:
        observations (th.ByteTensor): The observations of the episode.
        actions (th.LongTensor): The actions of the episode.
        rewards (th.FloatTensor): The rewards of the episode.
        ends (th.LongTensor): The episode ends of the episode.
        mask_padding (th.BoolTensor): The mask padding of the episode. True if the observation is padded, False otherwise.
    """
    observations: th.ByteTensor
    actions: th.LongTensor
    rewards: th.FloatTensor
    ends: th.LongTensor
    mask_padding: th.BoolTensor

    def __post_init__(self):
        for field in self.__dataclass_fields__.values():
            assert getattr(self, field.name).shape[0] == len(self)

        if self.ends.sum() > 0:
            idx_end = th.argmax(self.ends) + 1
            self.observations = self.observations[:idx_end]
            self.actions = self.actions[:idx_end]
            self.rewards = self.rewards[:idx_end]
            self.ends = self.ends[:idx_end]
            self.mask_padding = self.mask_padding[:idx_end]

    def __len__(self) -> int:
        return self.observations.size(0)

    def merge(self, other: "Episode") -> "Episode":
        """
        Merge two episodes into one.

        Args:
            other (Episode): The other episode to merge with.

        Returns:
            (Episode): The merged episode.
        """
        return Episode(
            th.cat((self.observations, other.observations), dim=0),
            th.cat((self.actions, other.actions), dim=0),
            th.cat((self.rewards, other.rewards), dim=0),
            th.cat((self.ends, other.ends), dim=0),
            th.cat((self.mask_padding, other.mask_padding), dim=0)
        )

    def segment(self, start: int, stop: int, should_pad: bool = False) -> "Episode":
        """
        Extract a segment of the episode.

        Args:
            start (int): The start index of the segment.
            stop (int): The stop index of the segment.
            should_pad (bool): If true, length of segment is stop - start. Otherwise, min(len(episode), stop) - max(0, start).

        Returns:
            (Episode): The extracted segment. If padding is required, the segment is padded.
        """
        assert start < len(self) and stop > 0 and start < stop
        # If padding is required and start / stop is given,
        # we extract a segment of the episode and pad it.
        padding_length_right = max(0, stop - len(self))  # only if stop > len(self)
        padding_length_left = max(0, -start)  # only if start < 0
        assert padding_length_left == padding_length_right == 0 or should_pad, \
            "If padding is required, the segment must be within the episode."

        def pad(x: th.Tensor):
            padded = x
            if padding_length_right:
                # 2 * x.ndim - 1 = [..., left(-2), right(-1)*]  (*: where padding is applied)
                # x_shape = [x_0, ... x_n] -> [x_0 + 2 * p_right, ..., x_n]
                padded = F.pad(x, pad=[0 for _ in range(2 * x.ndim - 1)] + [padding_length_right])
            if padding_length_left:
                # 2 * x.ndim - 2 = [..., left(-2)*, right(-1)]
                # x_shape = [x_0, ... x_n] -> [x_0 + 2 * p_left, ..., x_n]
                padded = F.pad(padded, pad=[0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0])

            return padded

        start = max(0, start)
        stop = min(len(self), stop)
        segment = Episode(self.observations[start:stop],
                          self.actions[start:stop],
                          self.rewards[start:stop],
                          self.ends[start:stop],
                          self.mask_padding[start:stop])

        segment.observations = pad(segment.observations)
        segment.actions = pad(segment.actions)
        segment.rewards = pad(segment.rewards)
        segment.ends = pad(segment.ends)
        segment.mask_padding = th.cat(
            [th.zeros(padding_length_left, dtype=th.bool),
             segment.mask_padding,
             th.zeros(padding_length_right, dtype=th.bool)],
            dim=0
        )  # because mask_padding is 1-dimensional

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(episode_length=len(self),
                              episode_return=self.rewards.sum())

    def save(self, path: Path):
        th.save(self.__dict__, path)
