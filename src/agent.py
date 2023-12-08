import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from pathlib import Path
from src.models.actor_critic import ActorCritic
from src.models.tokenizer import Tokenizer
from src.models.world_model import WorldModel
from src.utils import extract_state_dict


class Agent(nn.Module):
    """

    """
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self) -> th.device:
        return self.actor_critic.device

    def load(self,
             path_ckpt: Path,
             device: th.device,
             load_tokenizer: bool = True,
             load_world_model: bool = True,
             load_actor_critic: bool = True) -> None:
        """
        Load checkpoint of the agent (tokenizer, world model, actor critic).
        Args:
            path_ckpt: checkpoint path
            device: device to load the checkpoint
            load_tokenizer: whether to load the tokenizer
            load_world_model: whether to load the world model
            load_actor_critic: whether to load the actor critic
        """
        state_dict = th.load(path_ckpt, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(state_dict, 'actor_critic'))

    def get_action_token(self, obs: th.FloatTensor, sample: bool = True, temperature: float = 1.0) -> th.LongTensor:
        """
        Calculate the action token from the observation.

        1. Encode and decode the observation to get the reconstructed observation. (IRIS use reconstructed observation)
        2. Pass the reconstructed observation to the actor to get the logits of the action tokens.
        3. Sample or argmax (or Categorical dist) the logits to get the action token.

        Args:
            obs: current observation
            sample: weather to use sampling method or argmax method to select action token from logits
            temperature: temperature for sampling method. If high, the action token will be more random.

        Returns:
            action token (th.LongTensor): action token calculated from the logits return from the actor.
                1) w_a = Actor(hat{x}_t)
                2) token_a ~ Categorical(w_a) or token_a = argmax(w_a)
        """
        if self.actor_critic.use_real:
            input_obs = obs
        else:
            # Note that IRIS use reconstructed observation as input for actor and critic
            # input_obs: (b, c, H, W)
            input_obs = th.clamp(self.tokenizer.encode_decode(obs, preprocess=True, postprocess=True), 0, 1)

        logits_action = self.actor_critic.forward(input_obs).logits_actions[:, -1] / temperature   # (n, act_vocab_size)
        action_token = Categorical(logits=logits_action).sample() if sample else logits_action.argmax(dim=-1)   # (n, 1)
        return action_token
