import sys
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from dataclasses import dataclass
from einops import rearrange
from tqdm import tqdm

from src.envs.world_model_env import WorldModelEnv
from src.models.world_model import WorldModel
from src.models.tokenizer import Tokenizer
from src.dataset import Batch
from src.utils import calc_lambda_returns, LossWithIntermediateLosses

from typing import List, Optional, Tuple, Union


@dataclass
class ActorCriticOutput:
    """
    ActorCritic outputs

    Attributes:
        logits_actions (th.FloatTensor): (n, 1, act_vocab_size)
        mean_values (th.FloatTensor): (n, 1, 1)
    """
    logits_actions: th.FloatTensor
    mean_values: th.FloatTensor


@dataclass
class ImagineOutput:
    observations: th.ByteTensor
    actions: th.LongTensor
    logits_actions: th.FloatTensor
    values: th.FloatTensor
    rewards: th.FloatTensor
    ends: th.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size: int, use_real: bool = False):
        super().__init__()
        self.use_real = use_real

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_dim = 1024
        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(self.conv_dim, self.lstm_dim)
        self.hx, self.cx = None, None   # hidden states of LSTM

        self.critic_linear = nn.Linear(self.lstm_dim, 1)
        self.actor_linear = nn.Linear(self.lstm_dim, act_vocab_size)

    @property
    def device(self) -> th.device:
        return self.conv1.weight.device

    def clear(self) -> None:
        """
        Clear the hidden states of the LSTM to None.
        """
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_obs: Optional[th.Tensor] = None, mask_padding: Optional[th.Tensor] = None) -> None:
        """
        Reset the hidden states of the LSTM.
        If burnin_obs given, then, the hidden states are initialized with the LSTM's output for burnin_obs.

        Notes:
            burnin refers process of initializing the network with randomly initialized parameters (or burn-in values)
            for a few step to allow the network to stabilize before using it for actual training.

        Args:
            n (int): batch size = number of environments
            burnin_obs (Optional[th.Tensor]): (n, T, C, H, W) tensor of burn-in observations
            mask_padding (Optional[th.Tensor]): (n, T) tensor of mask for burn-in observations
        """
        device = self.conv1.weight.device
        self.hx = th.zeros(n, self.lstm_dim, device=device)
        self.cx = th.zeros(n, self.lstm_dim, device=device)

        if burnin_obs is not None:
            assert burnin_obs.ndim == 5 and burnin_obs.shape[0] == n, \
                f"ndim({burnin_obs.ndim}) != 5 and first shape({burnin_obs.size(0)}) != {n}"
            assert mask_padding is not None and burnin_obs.shape[:2] == mask_padding.shape, \
                f"shape({burnin_obs.shape[:2]}) != shape({mask_padding.shape})"

            for t in range(burnin_obs.size[1]):
                if mask_padding[:, t].any():
                    with th.no_grad():
                        self.forward(burnin_obs[:, t], mask_padding[:, t])

    def prune(self, mask: np.ndarray) -> None:
        """
        Prunes the hidden states of the LSTM where mask is False.
        """
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: th.FloatTensor, mask_padding: Optional[th.BoolTensor] = None) -> ActorCriticOutput:
        """
        Take an inputs as a single step of observation and fill the hidden states of the LSTM.
        Then, calculate the logits of actions and the mean values of the critic and return them.

        Args:
            inputs (th.FloatTensor): (n, C, H, W) tensor of observations
            mask_padding (th.BoolTensor): (n, ) tensor of mask for observations

        Returns:
            ActorCriticOutput: logits_actions (n, 1, act_vocab_size), mean_values (n, 1, 1)
        """
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64), \
            f"ndim({inputs.ndim}) != 4 and shape({inputs.shape[1:]}) != (3, 64, 64)"
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1, \
            f"all values of inputs should be in [0, 1], but min={inputs.min()} and max={inputs.max()}"
        assert mask_padding is None or mask_padding.ndim == 1 and mask_padding.shape[0] == inputs.shape[0], \
            f"ndim({mask_padding.ndim}) != 1 and shape({mask_padding.shape[0]}) != {inputs.shape[0]}"
        if mask_padding is not None:
            assert mask_padding.any(), "mask_padding should have at least one True value"

        # x: (n, C, H, W)
        x = inputs[mask_padding] if mask_padding is not None else inputs
        x = x.mul(2).sub(1)    # normalize to [-1, 1]

        x = F.relu(self.maxpool1(self.conv1(x)))   # (n, 32, 32, 32)
        x = F.relu(self.maxpool2(self.conv2(x)))   # (n, 32, 16, 16)
        x = F.relu(self.maxpool3(self.conv3(x)))   # (n, 64, 8, 8)
        x = F.relu(self.maxpool4(self.conv4(x)))   # (n, 64, 4, 4)
        x = th.flatten(x, start_dim=1)        # (n, 1024)
        assert x.shape[1] == self.conv_dim, f"shape({x.shape[1]}) != {self.conv_dim}"

        if mask_padding is None:
            # hx: (n, lstm_dim), cx: (n, lstm_dim)
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            out = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))
            self.hx[mask_padding], self.cx[mask_padding] = out

        logits_actions = self.actor_linear(self.hx)         # (n, act_vocab_size)
        mean_values = self.critic_linear(self.hx)           # (n, 1)
        logits_actions = rearrange(logits_actions, "n a -> n 1 a")
        mean_values = rearrange(mean_values, "n 1 -> n 1 1")

        return ActorCriticOutput(logits_actions, mean_values)

    def imagine(self,
                batch: Batch,
                tokenizer: Tokenizer,
                world_model: WorldModel,
                horizon: int,
                prog_bar: bool = False) -> ImagineOutput:
        """
        Create a sequence of observations, actions, logits of actions, values, rewards, and ends by imagining.

        Args:
            batch (Batch): Input batch, which contains observations and mask_padding.
            tokenizer (Tokenizer): Tokenizer for encoding and decoding observations.
            world_model (WorldModel): World model for predicting next observations.
            horizon (int): Horizon of imagination.
            prog_bar (bool): Whether to show progress bar.

        Returns:
            ImagineOutput: imagined outputs

                - observations (th.ByteTensor): (b, L, C, H, W)
                - actions (th.LongTensor): (b, L)
                - logits_actions (th.FloatTensor): (b, L, act_vocab_size)
                - values (th.FloatTensor): (b, L)
                - rewards (th.FloatTensor): (b, L)
                - ends (th.BoolTensor): (b, L)
        """
        assert not self.use_real, "original observations are not supported for imagination."

        init_obs = batch['observations']        # (b, T, C, H, W) = (b, T, 3, 64, 64)
        mask_padding = batch['mask_padding']    # (b, T)
        assert init_obs.ndim == 5 and init_obs.shape[2:] == (3, 64, 64), \
            f"ndim({init_obs.ndim}) != 5 and shape({init_obs.shape[2:]}) != (3, 64, 64)"
        assert mask_padding[:, -1].all(), "last observation should not be masked"

        device = init_obs.device
        world_model_env = WorldModelEnv(tokenizer, world_model, device)

        actions = []
        logits_actions = []
        values = []
        rewards = []
        ends = []
        observations = []

        burnin_obs = None
        if init_obs.shape[1] > 1:   # if T > 1
            burnin_obs = tokenizer.encode_decode(init_obs[:, :-1], preprocess=True, postprocess=True)
            burnin_obs = th.clamp(burnin_obs, 0, 1)     # (b, T-1, C, H, W)

        self.reset(n=init_obs.shape[0], burnin_obs=burnin_obs, mask_padding=mask_padding[:, :-1])

        obs = world_model_env.reset_from_initial_obs(init_obs[:, -1])   # (b, C, H, W)
        prod_bar = tqdm(range(horizon), disable=not prog_bar, desc="Imagine", file=sys.stdout)
        for t in prod_bar:
            observations.append(obs)
            ac_outputs = self.forward(obs)
            action_token = Categorical(logits=ac_outputs.logits_actions).sample()   # (b, 1)

            pred_next_obs = t < horizon - 1
            obs, reward, done, _ = world_model_env.step(action_token, pred_next_obs=pred_next_obs)

            actions.append(action_token)
            logits_actions.append(ac_outputs.logits_actions)
            values.append(ac_outputs.mean_values)
            rewards.append(th.tensor(reward).reshape(-1, 1))
            ends.append(th.tensor(done).reshape(-1, 1))

        self.clear()

        # L = horizon
        observations = th.stack(observations, dim=1).mul(255).byte()    # (b, L, C, H, W)
        actions = th.cat(actions, dim=1)                                # (b, L)
        logits_actions = th.cat(logits_actions, dim=1)                  # (b, L, act_vocab_size)
        values = th.cat(values, dim=1).squeeze(-1)                      # (b, L)
        rewards = th.cat(rewards, dim=1).to(device)                     # (b, L)
        ends = th.cat(ends, dim=1).to(device)                           # (b, L)

        return ImagineOutput(observations, actions, logits_actions, values, rewards, ends)

    def compute_loss(self,
                     batch: Batch,
                     tokenizer: Tokenizer,
                     world_model: WorldModel,
                     imag_horizon: int,
                     gamma: float,
                     lambda_: float,
                     ent_coef: float) -> LossWithIntermediateLosses:
        assert not self.use_real, "original observations are not supported for imagination."

        imag_outputs = self.imagine(batch, tokenizer, world_model, horizon=imag_horizon)

        # t = img_horizon
        with th.no_grad():
            lambda_returns = calc_lambda_returns(rewards=imag_outputs.rewards,
                                                 values=imag_outputs.values,
                                                 ends=imag_outputs.ends,
                                                 gamma=gamma,
                                                 lambda_=lambda_)[:, :-1]    # (b, t-1, 1)

        pi = Categorical(logits=imag_outputs.logits_actions[:, :-1])    # (b, t-1, act_vocab_size)
        log_probs = pi.log_prob(imag_outputs.actions[:, :-1])  # (b, t-1)
        loss_actions = -(log_probs * (lambda_returns - imag_outputs.values[:, :-1].detach())).mean()
        loss_entropy = -ent_coef * pi.entropy().mean()      # (b, t-1)
        loss_values = F.mse_loss(imag_outputs.values[:, :-1], lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions,
                                          loss_values=loss_values,
                                          loss_entropy=loss_entropy)