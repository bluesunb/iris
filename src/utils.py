import random, shutil
import cv2
import numpy as np
import torch as th
import torch.nn as nn

from collections import OrderedDict
from pathlib import Path

from src.episode import Episode
from typing import List, Tuple, Union, Optional, Dict, Any
from typing import OrderedDict as OrderedDictType


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def remove_dir(path: Path, should_ask: bool = False):
    assert path.is_dir()
    if not should_ask or input(f"Remove directory : {path} ? [Y/n]").lower() != 'n':
        shutil.rmtree(path)


def make_video(fname: str, fps: float, frames: np.ndarray) -> None:
    assert frames.ndim == 4, "Frames must have 4 dim: (timesteps, height, width, channels)"
    assert frames.shape[-1] == 3, "Frames must have 3 channels: RGB"
    height, width, channels = frames.shape[1:]
    video = cv2.VideoWriter(filename=str(fname),
                            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                            fps=fps, 
                            frameSize=(width, height))
    
    for frame in frames:
        video.write(frame[:, :, ::-1])  # BGR -> RGB
    video.release()


def init_weights(module: nn.Module) -> None:
    """
    Initialize the weights.
    `Linear` and `Embedding` layers are initialized with normal distribution with mean=0.0, std=0.02.
    `LayerNorm` layers are initialized with bias=0.0, weight=1.0.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict: Dict[str, Any], module_name: str) -> OrderedDictType[str, Any]:
    """
    Extract the state dict of the module with the given name.

    Args:
        state_dict (Dict[str, Any]): The state dict of the model.
        module_name (str): The name of the module to extract the state dict from.

    Returns:
        (OrderedDictType[str, Any]): The state dict of the module with the given name.
    """
    return OrderedDict(
        {k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)}
        # k = 'module.' + k
    )


def configure_optimizer(model: nn.Module, 
                        learning_rate: float, 
                        weight_decay: float,
                        *blacklist_names: List[str]) -> th.optim.AdamW:
    """
    Separate the model parameters into decay and no_decay groups through the blacklist_names and other conventions.
    Then return an AdamW optimizer with the decay and no_decay groups.

    Args:
        model (nn.Module): The model to configure the optimizer for.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        blacklist_names (List[str]): The names of the modules to exclude from weight decay.

    Returns:
        (th.optim.AdamW): The optimizer.
    """
    decay, no_decay = set(), set()
    whitelist_modules = (nn.Linear, nn.Conv2d)
    blacklist_modules = (nn.LayerNorm, nn.Embedding)

    # classify the parameters into decay and no_decay
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name
            
            if any([full_param_name.startswith(black_m_name) for black_m_name in blacklist_names]):
                # if parameters are in the blacklist names
                no_decay.add(full_param_name)
            elif 'bias' in param_name:
                # we exclude bias from weight decay
                no_decay.add(full_param_name)
            elif param_name.endswith("weight"):
                if isinstance(module, whitelist_modules):
                    decay.add(full_param_name)
                elif isinstance(module, blacklist_modules):
                    no_decay.add(full_param_name)

    param_dict = {param_name: param for param_name, param in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} should not appear in both decay and no_decay"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated into decay or no_decay"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
    ]

    optimizer = th.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def calc_lambda_returns(rewards: th.Tensor,
                        values: th.Tensor,
                        ends: th.Tensor,
                        gamma: float,
                        lambda_: float) -> th.Tensor:
    """
    Calculate the lambda returns for the given rewards, values, ends, gamma and lambda.
    $ L_t = r_t + \gamma(1 - d_t)[(1 - \lambda)V_{t+1} + \lambda L_{t+1}] $

    Args:
        rewards (th.Tensor): The rewards tensor of shape (bs, n_steps, 1).
        values (th.Tensor): The values tensor of shape (bs, n_steps, 1).
        ends (th.Tensor): The ends tensor of shape (bs, n_steps, 1).
        gamma (float): The discount factor.
        lambda_ (float): The lambda parameter.

    Returns:
        (np.ndarray): The lambda returns of shape (bs, n_steps, 1).
    """
    
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape} != {ends.shape} != {values.shape}"
    
    """
    To calculate the lambda returns, we seperate the calculation into non-recurrent and recurrent parts.

    $ L_t = r_t + \gamma(1 - d_t)[(1 - \lambda)V_{t+1} + \lambda L_{t+1}] $
          = r_t + \gamma(1 - d_t)(1 - \lambda)V_{t+1}   ... V_{t+1} is given by `values`
            + \gamma(1 - d_t)\lambda L_{t+1}            ... should be calculated recursively
    """

    lambda_return = th.empty_like(values)
    lambda_return[:, -1] = values[:, -1]    # L_T = V_T
    # L_t = r_t + gamma * (1 - d_t)  * (1 - m) * L_{t+1}
    lambda_return[:, :-1] = rewards[:, :-1] + gamma * ends[:, :-1].logical_not() * (1 - lambda_) * values[:, 1:]

    t = rewards.size(1)     # number of timesteps
    last_value = values[:, -1]
    for i in reversed(range(t - 1)):
        # L_i <- L_i + (1 - d_i) * gamma * lambda * (L_{i+1} - V_{i+1})
        lambda_return[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last_value
        last_value = lambda_return[:, i]

    return lambda_return


class LossWithIntermediateLosses:
    """
    Class to keep track of the individual losses for each component of the model, as well as the total loss.

    Attributes:
        loss_total (th.Tensor): Sum of all losses in the loss_dict
        intermediate_losses (Dict[str, Dict[str, th.Tensor]]): The dictionary of intermediate losses.
    """
    def __init__(self, **loss_dict: th.Tensor):
        """
        Args:
            loss_dict (Dict[str, th.Tensor]): The dictionary of losses.
        """
        self.loss_total = sum(loss_dict.values())
        self.intermediate_losses = {k: v.items() for k, v in loss_dict.items()}

    def __truediv__(self, value: float):
        """
        Apply the division to the total loss and all intermediate losses.
        """
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self
    

class EpisodeDirManager:
    """
    Class to save episode to specific directory.
    The episode is saved to the path: episode_dir/episode_{episode_id}_epoch_{epoch}.pth

    Attributes:
        episode_dir (Path):         The directory to save the episode to.
        max_num_episodes (int):     The maximum number of episodes to save.
        best_return (float):        The best return of the episodes saved. It use to save the best episode.
    """
    def __init__(self, episode_dir: Path, max_num_episodes: int):
        """
        Args:
            episode_dir (Path):         The directory to save the episode to.
            max_num_episodes (int):     The maximum number of episodes to save.
        """
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        """
        Save the episode to the episode directory to the path: episode_dir/episode_{episode_id}_epoch_{epoch}.pth
        """
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            
            episode_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith("episode_")]
            assert len(episode_paths) <= self.max_num_episodes, \
                f"Number of episodes exceeds the maximum number of episodes allowed: {self.max_num_episodes} < {len(episode_paths)}"
            
            # ====== if number of episodes to save is full ====== #
            if len(episode_paths) == self.max_num_episodes:
                # remove the episode with the smallest id, (p: episode_1_epoch_32.pth -> 1)
                to_remove = min(episode_paths, key=lambda p: int(p.stem.split('_')[1]))
                to_remove.unlink()
            
            episode.save(self.episode_dir / f"episode_{episode_id}_epoch_{epoch}.pth")

            # ====== check if the episode is the best episode ====== #
            episode_return = episode.compute_metrics().episode_return
            if episode_return > self.best_return:
                self.best_return = episode_return
                best_episode_path = [p for p in self.episode_dir.iterdir() if p.stem.startswith("best_")]
                assert len(best_episode_path) <= 1, f"Best episode path should be unique, but found {len(best_episode_path)} paths."
                
                if len(best_episode_path) == 1:
                    best_episode_path[0].unlink()

                episode.save(self.episode_dir / f"best_episode_{episode_id}_epoch_{epoch}.pth")


class RandomActionSelector:
    """
    Return random action.
    """
    def __init__(self, num_actions: int):
        """
        Args:
            num_actions (int): Total number of available actions.
        """
        self.num_actions = num_actions

    def select(self, obs: th.Tensor) -> th.Tensor:
        """
        Return random action with shape (obs.shape[0], ).
        Args:
            obs: observation to get batch size
        Returns:
            (th.Tensor): random action with shape (obs.shape[0], ) in (0 ~ num_actions - 1)
        """
        assert obs.ndim == 4, "Observation must have 4 dim: (bs, height, width, channels)"
        bs = obs.size(0)
        return th.randint(0, self.num_actions, size=(bs, ))
