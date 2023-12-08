import sys, time
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path

import gym
from tqdm import tqdm

import numpy as np
import torch as th
import torch.nn as nn

import wandb
import hydra
# from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.agent import Agent
from src.collector import Collector
from src.dataset import EpisodeDataset, EpisodeRamMonitoringDataset
from src.envs import SingleProcessEnv, MultiProcessEnv
from src.episode import Episode
from src.make_reconstruction import make_recon_from_batch
from src.models.actor_critic import ActorCritic
from src.models.world_model import WorldModel
from src.config import (Config, EnvCfg)
from src.utils import set_seed, configure_optimizer, EpisodeDirManager, LossWithIntermediateLosses
from src.default_configs.plater import instantiate

from typing import List, Dict, Any, Optional, Tuple, Union


def get_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    # def __init__(self, config: DictConfig):
    def __init__(self, config: Config):
        # wandb.init(config=OmegaConf.to_container(config, resolve=True),
        #            reinit=True, resume=True, **config.wandb.__dict__)

        if config.common.seed is not None:
            set_seed(config.common.seed)

        self.config = config
        self.start_epoch = 1
        self.device = th.device(config.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.recons_dir = self.media_dir / 'reconstructions'

        self.episode_mgr_imagine: EpisodeDirManager = None

        self.train_dataset: EpisodeRamMonitoringDataset = None
        self.test_dataset: EpisodeDataset = None
        self.train_collector: Collector = None
        self.test_collector: Collector = None

        if not config.common.resume:
            self.save_configs()

        env = self._setup()
        self._setup_optimizer(env)

        if config.initialization.path_ckpt is not None:
            self.agent.load(**config.initialization.__dict__, device=self.device)

        if config.common.resume:
            self.load_checkpoint()

    def _setup_optimizer(self, env: Union[SingleProcessEnv, MultiProcessEnv]) -> None:
        tokenizer = instantiate(self.config.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size,
                                 act_vocab_size=env.num_actions,
                                 config=instantiate(self.config.world_model))
        actor_critic = ActorCritic(**self.config.actor_critic.__dict__,
                                   act_vocab_size=env.num_actions)
        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)

        print(f"Number of parameters in tokenizer: {get_num_params(tokenizer)}")
        print(f"Number of parameters in world model: {get_num_params(world_model)}")
        print(f"Number of parameters in actor critic: {get_num_params(actor_critic)}")

        self.optimizer_tokenizer = th.optim.Adam(self.agent.tokenizer.parameters(),
                                                 lr=self.config.train.lr)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model,
                                                         learning_rate=self.config.train.lr,
                                                         weight_decay=self.config.train.world_model.weight_decay)
        self.optimizer_actor_critic = th.optim.Adam(self.agent.actor_critic.parameters(),
                                                    lr=self.config.train.lr)

    def _setup(self) -> Union[SingleProcessEnv, MultiProcessEnv]:
        assert self.config.train.should or self.config.eval.should, "Either train or eval should be True."
        episode_mgr_train = EpisodeDirManager(episode_dir=self.episode_dir / 'train',
                                              max_num_episodes=self.config.collection.train.num_episodes_to_save)
        episode_mgr_test = EpisodeDirManager(episode_dir=self.episode_dir / 'test',
                                             max_num_episodes=self.config.collection.test.num_episodes_to_save)
        self.episode_mgr_imagine = EpisodeDirManager(episode_dir=self.episode_dir / 'imagine',
                                                     max_num_episodes=self.config.eval.actor_critic.num_episodes_to_save)

        env = None
        if self.config.eval.should:
            test_env = self._create_env(self.config.env.test, num_envs=self.config.collection.test.num_envs)
            self.test_dataset = instantiate(self.config.dataset.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_mgr_test)
            env = test_env

        if self.config.train.should:
            train_env = self._create_env(self.config.env.train, num_envs=self.config.collection.train.num_envs)
            self.train_dataset = instantiate(self.config.dataset.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_mgr_train)
            env = train_env

        return env

    def save_configs(self) -> None:
        config_dir = Path('config')
        config_path = config_dir / 'trainer.yaml'
        config_dir.mkdir(exist_ok=True, parents=False)

        # shutil.copy('.hydra/config.yaml', config_path)
        # shutil.copytree(Path(hydra.utils.get_original_cwd()) / 'src', './src')
        # shutil.copytree(Path(hydra.utils.get_original_cwd()) / 'scripts', './scripts')

        # wandb.save(str(config_path))

        self.ckpt_dir.mkdir(exist_ok=True, parents=False)
        self.media_dir.mkdir(exist_ok=True, parents=False)
        self.episode_dir.mkdir(exist_ok=True, parents=False)

    @staticmethod
    def _create_env(cfg_env: EnvCfg, num_envs: int) -> Union[SingleProcessEnv, MultiProcessEnv]:
        env_fn = partial(instantiate, cfg=cfg_env)
        if num_envs > 1:
            return MultiProcessEnv(env_fn, num_envs, done_ratio_limit=1.0)
        return SingleProcessEnv(env_fn)

    def run(self) -> None:
        for epoch in range(self.start_epoch, self.config.common.epochs + 1):
            print(f"\nEpoch {epoch} / {self.config.common.epochs}\n")
            start_time = time.time()
            logs = []

            if self.config.train.should:
                if epoch <= self.config.collection.train.stop_after_epochs:
                    logs += self.train_collector.collect(agent=self.agent, epoch=epoch,
                                                         **self.config.collection.train.config.__dict__)

                logs += self.train_agent(epoch)

            if self.config.eval.should and (epoch % self.config.eval.eval_freq == 0):
                self.test_dataset.clear()  # clear the dataset
                logs += self.test_collector.collect(agent=self.agent, epoch=epoch,
                                                    **self.config.collection.test.config.__dict__)
                logs += self.eval_agent(epoch)

            if self.config.train.should:
                self.save_checkpoint(epoch, save_agent_only=not self.config.common.save_agent_only)

            logs.append({'duration': (time.time() - start_time) / 3600})
            # for metric in logs:
            #     wandb.log({'epoch': epoch, **metric})

            print(f"Epoch {epoch} took {(time.time() - start_time) / 3600:.2f} hours.")

        print("Training finished.")
        # wandb.finish()

    def train_agent(self, epoch: int) -> List[Dict[str, int]]:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer = defaultdict(float)
        metrics_world_model = defaultdict(float)
        metrics_actor_critic = defaultdict(float)

        config_tokenizer = self.config.train.tokenizer
        config_world_model = self.config.train.world_model
        config_actor_critic = self.config.train.actor_critic

        if epoch > config_tokenizer.start_after_epochs:
            metrics_tokenizer = self.train_component(self.agent.tokenizer,
                                                     self.optimizer_tokenizer,
                                                     seq_len=1,
                                                     sample_from_start=True,
                                                     **config_tokenizer.__dict__)
            self.agent.tokenizer.eval()

        if epoch > config_world_model.start_after_epochs:
            metrics_world_model = self.train_component(self.agent.world_model,
                                                       self.optimizer_world_model,
                                                       seq_len=self.config.common.seq_len,
                                                       sample_from_start=True,
                                                       tokenizer=self.agent.tokenizer,
                                                       **config_world_model.__dict__)
            self.agent.world_model.eval()

        if epoch > config_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component(self.agent.actor_critic,
                                                        self.optimizer_tokenizer,
                                                        seq_len=1 + self.config.train.actor_critic.burnin_steps,
                                                        sample_from_start=False,
                                                        tokenizers=self.agent.tokenizer,
                                                        world_model=self.agent.world_model,
                                                        **config_actor_critic.__dict__)

        return [{'epoch': epoch,
                 **metrics_tokenizer,
                 **metrics_world_model,
                 **metrics_actor_critic}]

    @th.no_grad()
    def eval_agent(self, epoch: int) -> List[Union[Dict[str, float], Dict[Any, float]]]:
        self.agent.eval()

        metrics_tokenizer = defaultdict(float)
        metrics_world_model = defaultdict(float)

        config_tokenizer = self.config.eval.tokenizer
        config_world_model = self.config.eval.world_model
        config_actor_critic = self.config.eval.actor_critic

        if epoch > config_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer,
                                                    batch_num_samples=config_tokenizer.batch_num_samples,
                                                    seq_len=1)

        if epoch > config_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(self.agent.world_model,
                                                      config_world_model.batch_num_samples,
                                                      seq_len=self.config.common.seq_len,
                                                      tokenizer=self.agent.tokenizer)

        if epoch > config_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if config_tokenizer.save_reconstructions:
            batch = self.test_dataset.sample_batch(batch_num_samples=3,
                                                   seq_len=self.config.common.seq_len)

            make_recon_from_batch(batch, save_dir=self.recons_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    def train_component(self,
                        component: nn.Module,
                        optimizer: th.optim.Optimizer,
                        steps_per_epochs: int,
                        batch_num_samples: int,
                        grad_acc_steps: int,
                        max_grad_norm: Optional[float],
                        seq_len: int,
                        sample_from_start: bool,
                        **loss_kwargs: Any) -> Dict[str, float]:
        """
        Trains a component (tokenizer, world model, actor critic) for one epoch.

        Args:
            component (nn.Module): component to train
            optimizer (th.optim.Optimizer): optimizer for the component
            steps_per_epochs (int): number of steps per epoch
            batch_num_samples (int): number of samples per batch
            grad_acc_steps (int): gradient accumulation steps
            max_grad_norm (Optional[float]): maximum gradient norm
            seq_len (int): The length of each episode segment in the batch
            sample_from_start (bool): whether to sample from the start of the episode or not
            **loss_kwargs:

        Returns:
            Dict[str, float]: losses for train and eval of the component
        """

        total_loss = 0.0
        intermediate_losses = defaultdict(float)

        assert len(f"{component}") < 100, f"Component name is too long."  # TODO: remove this
        prog_bar = tqdm(range(steps_per_epochs), desc=f"Training {component}", file=sys.stdout)
        for _ in prog_bar:
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_dataset.sample_batch(batch_num_samples, seq_len, sample_from_start)
                batch = self._to_device(batch)

                # compute loss
                losses: LossWithIntermediateLosses = component.compute_loss(batch, **loss_kwargs) / grad_acc_steps
                step_loss = losses.loss_total
                step_loss.backward()
                total_loss += step_loss.item() / steps_per_epochs

                # accumulate intermediate losses
                for loss_name, loss in losses.intermediate_losses.items():
                    intermediate_losses[f"{component}/train/{loss_name}"] += loss / steps_per_epochs

            # clip gradients
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm(component.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {f"{component}/train/total_loss": total_loss, **intermediate_losses}
        return metrics

    def inspect_imagination(self, epoch: int) -> List[dict]:
        mode_name = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_mgr_imagine.max_num_episodes,
                                               seq_len=1 + self.config.train.actor_critic.burnin_steps,
                                               sample_from_start=False)
        batch = self._to_device(batch)
        outputs = self.agent.actor_critic.imagine(batch,
                                                  tokenizer=self.agent.tokenizer,
                                                  world_model=self.agent.world_model,
                                                  horizon=self.config.eval.actor_critic.horizon,
                                                  prog_bar=True)

        logs = []
        traj_generator = zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(),
                             outputs.ends.cpu())
        for i, (o, a, r, d) in enumerate(traj_generator):
            episode = Episode(observations=o, actions=a, rewards=r, ends=d, mask_padding=th.ones_like(d))
            episode_id = (epoch - 1 - self.config.train.actor_critic.start_after_epochs) * outputs.observations.size(
                0) + i
            self.episode_mgr_imagine.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            # metrics_episode['action_histogram'] = wandb.Histogram(sequence=episode.actions.numpy(),
            #                                                       num_bins=self.agent.world_model.act_vocab_size)
            logs.append({f"{mode_name}/{k}": v for k, v in metrics_episode.items()})

        return logs

    @th.no_grad()
    def eval_component(self,
                       component: nn.Module,
                       batch_num_samples: int,
                       seq_len: int,
                       **loss_kwargs: Any) -> Dict[str, float]:

        total_loss = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        prog_bar = tqdm(desc=f"Evaluating {component}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, chunk_size=seq_len):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **loss_kwargs)
            total_loss += losses.loss_total.item()

            for loss_name, loss in losses.intermediate_losses.items():
                intermediate_losses[f"{component}/eval/{loss_name}"] += loss

            steps += 1
            prog_bar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f"{component}/eval/total_loss": total_loss / steps, **intermediate_losses}
        return metrics

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        """
        Save checkpoint of the agent (tokenizer, world model, actor critic).
        If not `save_agent_only`, also save the epoch, optimizer, and dataset.

        Args:
            epoch (int): current epoch
            save_agent_only (int): whether to save only the agent or not
        """
        # save agent
        th.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')

        if not save_agent_only:
            # save epoch
            th.save(epoch, self.ckpt_dir / 'epoch.pt')
            # save optimizer
            th.save({"optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                     "optimizer_world_model": self.optimizer_world_model.state_dict(),
                     "optimizer_actor_critic": self.optimizer_actor_critic.state_dict()},
                    self.ckpt_dir / 'optimizer.pt')

            # save dataset
            dataset_ckpt_dir = self.ckpt_dir / 'dataset'
            dataset_ckpt_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(dataset_ckpt_dir)

            if self.config.eval.should:
                # save test dataset
                th.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_ckpt_dir = Path("checkpoints_tmp")
        shutil.copytree(self.ckpt_dir, tmp_ckpt_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_ckpt_dir)

    def load_checkpoints(self) -> None:
        assert self.ckpt_dir.is_dir(), f"Checkpoint directory {self.ckpt_dir} does not exist."
        self.start_epoch = th.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)

        ckpt_optimizer = th.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_optimizer['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_optimizer['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_optimizer['optimizer_actor_critic'])

        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.config.eval.should:
            self.test_dataset.num_seen_episodes = th.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

        print(f"Loaded checkpoint from {self.ckpt_dir.absolute()}.")
        print(f"Total {len(self.train_dataset)} episodes in the train dataset.")

    def _to_device(self, batch: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}


if __name__ == "__main__":
    config = Config()

    config.collection.test.num_envs = 2
    config.collection.train.config.burnin_steps = 5
    config.tokenizer.emb_dim = 360
    config.world_model.n_layers = 3

    config.tokenizer.encoder.config.z_channels = 128
    config.tokenizer.decoder.config.z_channels = 128
    config.tokenizer.encoder.config.channel = 8
    config.tokenizer.vocab_size = 100
    config.tokenizer.emb_dim = 72

    config.collection.train.config.max_steps = 100

    config.world_model.emb_dim = 84

    trainer = Trainer(config)
    trainer.run()
