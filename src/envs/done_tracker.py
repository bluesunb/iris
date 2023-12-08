import numpy as np


class DoneTrackerEnv:
    """
    Monitor the environment's done status.
    - 0: not done
    - 1: done
    - 2: already done
    """
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.done_tracker: np.zeros(self.num_envs, dtype=np.uint8)
    
    def reset_done_tracker(self) -> None:
        self.done_tracker = np.zeros(self.num_envs, dtype=np.uint8)
    
    def update_done_tracker(self, done: np.ndarray) -> None:
        """
        Apply the done status to the done tracker and evaluate the done status.
        If done_tracker == 1 -> done_tracker = 2
        Then apply the done status to the done tracker.
        """
        self.done_tracker = np.clip(2 * self.done_tracker + done, 0, 2)
    
    @property
    def num_envs_done(self) -> int:
        """
        number of doned environments
        """
        return (self.done_tracker > 0).sum()
    
    @property
    def mask_ongoing(self) -> np.ndarray:
        """
        mask to true for ongoing environments
        """
        return np.logical_not(self.done_tracker)
    
    @property
    def mask_new_dones(self) -> np.ndarray:
        """
        mask to true for onging envirnments which are just done or onging, not already done
        """
        return np.logical_not(self.done_tracker[self.done_tracker < 2])