import sys, random
import wandb
import numpy as np
import torch as th

from tqdm import tqdm
from einops import rearrange
from typing import List, Optional, Union, Tuple, Dict

from src.agent import Agent
from src.dataset import EpisodeDataset
from src.envs import SingleProcessEnv, MultiProcessEnv
from src.episode import Episode
from src.utils import EpisodeDirManager, RandomActionSelector


class Collector:
    def __init__(self,
                 env: Union[SingleProcessEnv, MultiProcessEnv],
                 dataset: EpisodeDataset,
                 episode_dir_manager: EpisodeDirManager):
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.obs = self.env.reset()  # (num_envs, c, h, w)
        self.episode_ids: List[Optional[int]] = [None] * self.env.num_envs
        self.action_selector = RandomActionSelector(self.env.num_actions)

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.episode_ids = [None] * self.env.num_envs

    @th.no_grad()
    def collect(self,
                agent: Agent,
                epoch: int,
                epsilon: float,
                sample: bool,
                temperature: float,
                burnin_steps: int,
                *,
                max_steps: Optional[int] = None,
                max_episodes: Optional[int] = None) -> List[Dict[str, Union[int, float]]]:
        assert self.env.num_actions == agent.world_model.act_vocab_size, \
            f"env.num_action ({self.env.num_actions}) != world model.act_vocab_size ({agent.world_model.act_vocab_size})"
        assert 0 <= epsilon <= 1, f"epsilon ({epsilon}) must be in [0, 1]"
        assert (max_steps is None) != (max_episodes is None), \
            f"Either num_steps ({max_steps}) or num_episodes ({max_episodes}) must be specified, but not both"

        logs = []
        n_steps, n_episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        # burnin_obs_recon, mask_padding = None, None
        # if set(self.episode_ids) != {None} and burnin_steps > 0:     # if we have to burn in,
        #     segmented_episodes = []
        #     for episode_id in self.episode_ids:
        #         episode = self.dataset.get_episode(episode_id)
        #         seg_episode = episode.segment(start=len(episode) - burnin_steps, stop=len(episode), should_pad=True)
        #         segmented_episodes.append(seg_episode)

        #     mask_padding = th.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
        #     burnin_obs = th.stack([episode.observations for episode in segmented_episodes], dim=0)
        #     burnin_obs = burnin_obs.float().div(255).to(agent.device)
        #     burnin_obs_recon = th.clamp(agent.tokenizer.encode_decode(burnin_obs, preprocess=True, postprocess=True), 0, 1)

        # agent.actor_critic.reset(n=self.env.num_envs, burnin_obs=burnin_obs_recon, mask_padding=mask_padding)
        self.burnin(agent, burnin_steps)

        # ======= Collect experience =======
        prog_bar = tqdm(total=max_steps or max_episodes, desc=f"Experience Collect ({self.dataset.name})",
                        file=sys.stdout)

        def should_stop(n_steps: int, n_episodes: int) -> bool:
            if max_steps is not None:
                return n_steps >= max_steps
            return n_episodes >= max_episodes

        while not should_stop(n_steps, n_episodes):
            observations.append(self.obs)
            obs = rearrange(th.FloatTensor(self.obs).div(255), 'n h w c -> n c h w').to(agent.device)

            if random.random() < epsilon:   # epsilon greedy
                action = self.action_selector.select(obs).cpu().numpy()
            else:
                action = agent.get_action_token(obs, sample=sample, temperature=temperature).cpu().numpy()

            self.obs, reward, done, info = self.env.step(action)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            # true for ongoing envs which are just done or ongoing, not already done
            new_steps = len(self.env.mask_new_dones)    # calc ongoing envs
            n_steps += new_steps
            prog_bar.update(0 or new_steps)

            if self.env.should_reset():
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                new_episodes = self.env.num_envs
                n_episodes += new_episodes
                prog_bar.update(0 or new_episodes)

                epoch_return, epoch_log = self.save_and_log_episodes(epoch)
                returns.extend(epoch_return)
                logs.extend(epoch_log)

                self.reset()
                agent.actor_critic.reset(n=self.env.num_envs)
                observations, actions, rewards, dones = [], [], [], []

        if len(observations) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones)
        agent.actor_critic.clear()
        mterics_collect = {'#episodes': len(self.dataset),
                           '#steps': sum(map(len, self.dataset.episodes))}
        if len(returns) > 0:
            mterics_collect['return'] = np.mean(returns)
        mterics_collect = {f"{self.dataset.name}/{k}": v for k, v in mterics_collect.items()}
        logs.append(mterics_collect)

        return logs

    def save_and_log_episodes(self, epoch: int) -> Tuple[List[float], List[Dict[str, Union[int, float]]]]:
        returns, log = [], []
        for episode_id in self.episode_ids:
            episode = self.dataset.get_episode(episode_id)
            self.episode_dir_manager.save(episode, episode_id, epoch=epoch)
            metrics_episode = episode.compute_metrics().__dict__.copy()
            metrics_episode['episode_num'] = episode_id

            np_histogram = np.histogram(episode.actions.numpy(),
                                        # position of the bins' left edges
                                        bins=np.arange(self.env.num_actions + 1) - 0.5,
                                        density=True)
            metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np_histogram)
            log.append({f"{self.dataset.name}/{k}": v for k, v in metrics_episode.items()})
            returns.append(metrics_episode['episode_return'])

        return returns, log

    def burnin(self, agent: Agent, burnin_steps: int) -> None:
        """
        Make burnin obs and reset the actor_critic.
        """
        burnin_obs_recon, mask_padding = None, None
        if set(self.episode_ids) != {None} and burnin_steps > 0:  # if we have to burn in,
            segmented_episodes = []
            for episode_id in self.episode_ids:
                episode = self.dataset.get_episode(episode_id)
                seg_episode = episode.segment(start=len(episode) - burnin_steps, stop=len(episode), should_pad=True)
                segmented_episodes.append(seg_episode)

            mask_padding = th.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
            burnin_obs = th.stack([episode.observations for episode in segmented_episodes], dim=0)
            burnin_obs = burnin_obs.float().div(255).to(agent.device)
            burnin_obs_recon = th.clamp(agent.tokenizer.encode_decode(burnin_obs, preprocess=True, postprocess=True), 
                                        0, 1)

        agent.actor_critic.reset(n=self.env.num_envs, burnin_obs=burnin_obs_recon, mask_padding=mask_padding)

    def add_experience_to_dataset(self, *args) -> None:
        """
        From given obs, actions, rewards and ends, add this sequence of experience to the dataset.

        Args:
            *args: Observations, actions, rewards, dones
        """
        assert len(set(map(len, args))) == 1, "All arguments must have the same length"
        for i, (o, a, r, d) in enumerate(zip(*map(lambda x: np.swapaxes(x, 0, 1), args))):
            # o: (num_env, bs, c, h, w), a: (num_env, bs), r: (num_env, bs), d: (num_env, bs)
            # -> (bs, num_env, ...)
            episode = Episode(
                observations=th.ByteTensor(o).permute(0, 3, 1, 2).contiguous(), # (t, h, w, c) -> (t, c, h, w)
                actions=th.LongTensor(a),   # (t, )
                rewards=th.FloatTensor(r),  # (t, )
                ends=th.LongTensor(d),      # (t, )
                mask_padding=th.ones(d.shape[0], dtype=th.bool)
            )
            if self.episode_ids[i] is None:
                self.episode_ids[i] = self.dataset.add_episode(episode)
            else:
                self.dataset.update_episode(self.episode_ids[i], episode)
