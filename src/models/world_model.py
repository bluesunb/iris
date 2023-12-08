import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from einops import rearrange
from typing import Optional, Tuple, List, Any, Dict

from sympy.polys.polyoptions import Auto

from src.dataset import Batch
from src.models.kv_caching import KeysValues
from src.models.slicer import Embedder, Head
from src.models.tokenizer import Tokenizer
from src.models.transformer import Transformer, TransformerConfig
from src.utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    """
    Attributes:
        output_sequence (th.FloatTensor): (bs, t, e), transformer output sequence.
        logits_observations (th.FloatTensor): (bs, t, obs_vocab_size), logits for observations.
        logits_rewards (th.FloatTensor): (bs, t, 3), logits for rewards.
        logits_ends (th.FloatTensor): (bs, t, 2), logits for ends.
    """
    output_sequence: th.FloatTensor
    logits_observations: th.FloatTensor
    logits_rewards: th.FloatTensor
    logits_ends: th.FloatTensor


class WorldModel(nn.Module):
    """
    It predicts the next observation, reward and end given the current observation and action.
    """
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig):
        super().__init__()
        self.obs_vocab_size = obs_vocab_size    # tokenizer  vocab size
        self.act_vocab_size = act_vocab_size    # num actions
        self.config = config
        self.transformer = Transformer(config)

        # token indication: [obs, ..., last_obs, action] (len = tokens_per_block)
        nonlast_obs_token_mask = th.ones(self.config.tokens_per_block)
        nonlast_obs_token_mask[-2] = 0  # [1, ..., 0, 1]
        act_token_mask = th.zeros(self.config.tokens_per_block)
        act_token_mask[-1] = 1  # [0, ..., 0, 1]
        obs_token_mask = 1 - act_token_mask  # [1, ..., 1, 0]

        self.pos_emb = nn.Embedding(self.config.max_tokens, self.config.emb_dim)
        self.embedder = Embedder(
            max_blocks=self.config.max_blocks,
            block_masks=[act_token_mask, obs_token_mask],
            embed_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.emb_dim),
                                        nn.Embedding(obs_vocab_size, config.emb_dim)])
        )

        self.head_observations = Head(max_blocks=config.max_blocks,
                                      block_masks=nonlast_obs_token_mask,
                                      head_module=self.get_head_module(obs_vocab_size))

        self.head_rewards = Head(max_blocks=config.max_blocks,
                                 block_masks=act_token_mask,
                                 head_module=self.get_head_module(3))  # 3 rewards: -1, 0, 1

        self.head_ends = Head(max_blocks=config.max_blocks,
                              block_masks=act_token_mask,
                              head_module=self.get_head_module(2))  # 2 ends: 0, 1

        self.apply(init_weights)

    def get_head_module(self, out_dim: int):
        return nn.Sequential(nn.Linear(self.config.emb_dim, self.config.emb_dim),
                             nn.ReLU(),
                             nn.Linear(self.config.emb_dim, out_dim))

    def forward(self, tokens: th.LongTensor, prev_kvs: Optional[KeysValues] = None) -> WorldModelOutput:
        """
        Compute the prediction of the obs, rewards, ends at the next time step given the current tokens.
        Args:
            tokens (th.LongTensor): (bs, t)
            prev_kvs (Optional[KeysValues]): (bs, t, e) or None, previous key-value pairs saved in the cache.

        Returns:
            WorldModelOutput: (output_sequence, logits_observations, logits_rewards, logits_ends)

            - output_sequence: (bs, t, e), transformer output sequence.
            - logits_observations: (bs, t, obs_vocab_size), logits for observations.
            - logits_rewards: (bs, t, 3), logits for rewards.
            - logits_ends: (bs, t, 2), logits for ends.
        """
        n_steps = tokens.size(1)
        assert n_steps <= self.config.max_tokens, f"number of tokens({n_steps}) must be <= max_tokens({self.config.max_tokens})"
        prev_steps = 0 if prev_kvs is None else prev_kvs.size

        sequences = self.embedder(tokens, n_steps, prev_steps)  # (bs, t, tf_e)
        # Add positional embedding
        sequences = sequences + self.pos_emb(prev_steps + th.arange(n_steps, device=tokens.device))
        # compute next latent by transformer
        x = self.transformer(sequences, prev_kvs)       # (bs, t, tf_e)

        logits_observations = self.head_observations(x, n_steps, prev_steps)    # (bs, t, obs_vocab_size)
        logits_rewards = self.head_rewards(x, n_steps, prev_steps)            # (bs, t, 3)
        logits_ends = self.head_ends(x, n_steps, prev_steps)                # (bs, t, 2)

        return WorldModelOutput(output_sequence=x,
                                logits_observations=logits_observations,
                                logits_rewards=logits_rewards,
                                logits_ends=logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer) -> LossWithIntermediateLosses:
        """
        Compute the loss of the world model by comparing the model output (preiction) and the label sequence of the given batch.

        Args:
            batch (Batch): batch of the dataset. It must have "observations", "action", "rewards", "ends" and "mask_padding".
            tokenizer (Tokenizer): tokenizer for encoding the observation sequence.

        Returns:
            LossWithIntermediateLosses: loss_obs, loss_rewards, loss_ends
        """
        with th.no_grad():
            # obs_tokens: (bs, t, hw)
            obs_tokens = tokenizer.encode(batch["observations"], preprocess=True).tokens

        # action_tokens: (bs, t, 1)
        action_tokens = rearrange(batch["actions"], 'bs t -> bs t 1')
        # tokens: (bs, t(hw+1))
        tokens = rearrange(th.cat((obs_tokens, action_tokens), dim=2), 'bs t k -> bs (t k)')
        outputs = self.forward(tokens)      # (bs, t, e)

        # compute loss
        labels_observations, labels_rewards, labels_ends = self.compute_labels(obs_tokens,
                                                                               batch["rewards"],
                                                                               batch["ends"],
                                                                               batch["mask_padding"])

        # learn dynamics
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        logits_rewards = rearrange(outputs.logits_rewards, 'b t e -> (b t) e')
        logits_ends = rearrange(outputs.logits_ends, 'b t e -> (b t) e')

        # loss_obs = F.cross_entropy(logits_observations, labels_observations, ignore_index=-100)
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(logits_rewards, labels_rewards)
        loss_ends = F.cross_entropy(logits_ends, labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs,
                                          loss_rewards=loss_rewards,
                                          loss_ends=loss_ends)

    def compute_labels(self,
                       obs_tokens: th.Tensor,
                       rewards: th.Tensor,
                       ends: th.Tensor,
                       mask_padding: th.BoolTensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Mask the given batch sequence with mask_padding and return them as labels.

        Args:
            obs_tokens (th.Tensor): (bs, t, hw), observation tokens. Note that the token is argmin index of z.
            rewards (th.Tensor): (bs, t)    -1: fail, 0: nothing, 1: success
            ends (th.Tensor): (bs, t)    0: not done, 1: done
            mask_padding (th.BoolTensor): (bs, t)   True: not padding, False: padding

        Returns:
            Tuple[th.Tensor, th.Tensor, th.Tensor]: (labels_observations, labels_rewards, labels_ends)

        """
        assert th.all(ends.sum(dim=1) <= 1), "ends must have at most one 1(done) in each sequence"
        mask_fill = th.logical_not(mask_padding)

        # mask each given batch sequence where the mask_fill is True
        labels_observations = obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100)
        labels_observations = rearrange(labels_observations, 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.flatten(), labels_rewards.flatten(), labels_ends.flatten()

    def __repr__(self):
        return (f"WorldModel(obs_vocab_size={self.obs_vocab_size}, act_vocab_size={self.act_vocab_size}, "
                f"Trsf={self.transformer})")

    def __str__(self):
        return super().__repr__()


if __name__ == "__main__":
    from src.models.tokenizer import Tokenizer, AutoencoderConfig, Encoder, Decoder

    tf_config = TransformerConfig(tokens_per_block=16 * 16 + 1,  # obs + act
                                  max_blocks=5,
                                  attention='block_casual',
                                  n_layers=2,
                                  n_heads=4,
                                  emb_dim=128,
                                  emb_dropout=0.0,
                                  residual_dropout=0.0,
                                  attn_dropout=0.0)

    ae_config = AutoencoderConfig(resolution=64,
                                  in_channels=3,
                                  z_channels=128,
                                  out_channels=24,
                                  channel=32,
                                  channel_scales=(2, 3, 4),
                                  num_res_blocks=2,
                                  attn_resolutions=(128,),
                                  dropout=0.0)

    obs_vocab_size = 10
    act_vocab_size = 19

    encoder = Encoder(ae_config)
    decoder = Decoder(ae_config)
    tokenizer = Tokenizer(vocab_size=obs_vocab_size, emb_dim=144, encoder=encoder, decoder=decoder, lpips=False)

    world_model = WorldModel(obs_vocab_size, act_vocab_size, tf_config)

    batch_size, seq_len = 3, 5
    c, h, w = 3, 64, 64

    ends = th.zeros(batch_size, seq_len)
    ends[:, -1] = 1
    mask_padding = th.multinomial(th.tensor([1., 5.]), batch_size * seq_len, replacement=True).bool().view(batch_size, seq_len)
    batch = dict(observations=th.randn(batch_size, seq_len, c, h, w),
                 actions=th.randint(0, 4, (batch_size, seq_len)),
                 rewards=th.randint(-1, 2, (batch_size, seq_len)),
                 ends=ends,
                 mask_padding=mask_padding)

    loss = world_model.compute_loss(batch, tokenizer)
