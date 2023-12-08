import torch as th

from dataclasses import dataclass
from src.default_configs import ActorCriticCfg, DatasetCfg, EnvCfg, TokenizerCfg, WorldModelCfg
from typing import List, Dict, Any, Optional, Tuple, Union
from src.default_configs.plater import HydraWrapper


@dataclass
class WandbCfg(HydraWrapper):
    mode: str = "online"
    project: str = "iris"
    entity = None
    name = None
    group = None
    tags = None
    notes = None


@dataclass
class InitializationCfg(HydraWrapper):
    path_ckpt: bool = None
    load_tokenize: bool = False
    load_world_model: bool = False
    load_actor_critic: bool = False


@dataclass
class CommonCfg(HydraWrapper):
    epochs: int = 600
    device: str = "cuda:0"
    save_agent_only: bool = True
    seed: int = 0
    seq_len: int = WorldModelCfg.max_blocks
    resume: bool = False


@dataclass
class TokenizerTrainCfg(HydraWrapper):
    batch_num_samples: int = 256
    grad_acc_steps: int = 1
    max_grad_norm: float = 10.0
    start_after_epochs: int = 5
    steps_per_epoch: int = 200


@dataclass
class WorldModelTrainCfg(HydraWrapper):
    batch_num_samples: int = 64
    grad_acc_steps: int = 1
    max_grad_norm: float = 10.0
    weight_decay: float = 0.01
    start_after_epochs: int = 25
    steps_per_epoch: int = 200


@dataclass
class ActorCriticTrainCfg(HydraWrapper):
    batch_num_samples: int = 64
    grad_acc_steps: int = 1
    max_grad_norm: float = 10.0
    start_after_epochs: int = 50
    steps_per_epoch: int = 200
    imagine_horizon: int = CommonCfg.seq_len
    burnin_steps: int = 20
    gamma: float = 0.995
    lambda_: float = 0.95
    entropy_weight: float = 0.001


@dataclass
class TrainCfg(HydraWrapper):
    should: bool = True
    lr: float = 0.0001
    tokenizer: TokenizerTrainCfg = TokenizerTrainCfg()
    world_model: WorldModelTrainCfg = WorldModelTrainCfg()
    actor_critic: ActorCriticTrainCfg = ActorCriticTrainCfg()


@dataclass
class TokenizerEvalCfg(HydraWrapper):
    batch_num_samples: int = TrainCfg.tokenizer.batch_num_samples
    start_after_epochs: int = TrainCfg.tokenizer.start_after_epochs
    save_reconstructions: bool = True


@dataclass
class WorldModelEvalCfg(HydraWrapper):
    batch_num_samples: int = TrainCfg.world_model.batch_num_samples
    start_after_epochs: int = TrainCfg.world_model.start_after_epochs


@dataclass
class ActorCriticEvalCfg(HydraWrapper):
    num_episodes_to_save: int = TrainCfg.actor_critic.batch_num_samples
    horizon: int = TrainCfg.actor_critic.imagine_horizon
    start_after_epochs: int = TrainCfg.actor_critic.start_after_epochs


@dataclass
class EvalCfg(HydraWrapper):
    should: bool = True
    eval_freq: int = 5
    tokenizer: TokenizerEvalCfg = TokenizerEvalCfg()
    world_model: WorldModelEvalCfg = WorldModelEvalCfg()
    actor_critic: ActorCriticEvalCfg = ActorCriticEvalCfg()


@dataclass
class EpisodeCollectCfg(HydraWrapper):
    epsilon: float = 0.01
    sample: bool = True
    temperature: float = 1.0
    burnin_steps: int = ActorCriticTrainCfg.burnin_steps


@dataclass
class EpisodeCollectTrainCfg(EpisodeCollectCfg):
    max_steps: int = 200


@dataclass
class EpisodeCollectTestCfg(EpisodeCollectCfg):
    max_episodes: int = 16


@dataclass
class CollectionTrainCfg(HydraWrapper):
    num_envs: int = 1
    stop_after_epochs: int = 500
    num_episodes_to_save: int = 10
    config: EpisodeCollectTrainCfg = EpisodeCollectTrainCfg(epsilon=0.01,
                                                            sample=True,
                                                            temperature=1.0,
                                                            max_steps=200,
                                                            burnin_steps=ActorCriticTrainCfg.burnin_steps)


@dataclass
class CollectionTestCfg(HydraWrapper):
    num_envs: int = 8
    num_episodes_to_save: int = CollectionTrainCfg.num_episodes_to_save
    config: EpisodeCollectTestCfg = EpisodeCollectTestCfg(epsilon=0.0,
                                                          sample=True,
                                                          temperature=0.5,
                                                          max_episodes=200,
                                                          burnin_steps=ActorCriticTrainCfg.burnin_steps)


@dataclass
class CollectionCfg(HydraWrapper):
    train: CollectionTrainCfg = CollectionTrainCfg()
    test: CollectionTestCfg = CollectionTestCfg()

    def __post_init__(self):
        self.test.num_episodes_to_save = self.train.num_episodes_to_save


@dataclass
class Config(HydraWrapper):
    common: CommonCfg = CommonCfg()
    env: EnvCfg = EnvCfg()
    dataset: DatasetCfg = DatasetCfg()
    wandb: WandbCfg = WandbCfg()
    initialization: InitializationCfg = InitializationCfg()
    collection: CollectionCfg = CollectionCfg()
    train: TrainCfg = TrainCfg()
    eval: EvalCfg = EvalCfg()
    tokenizer: TokenizerCfg = TokenizerCfg()
    world_model: WorldModelCfg = WorldModelCfg()
    actor_critic: ActorCriticCfg = ActorCriticCfg()

    def __post_init__(self):
        self.common.seq_len = self.world_model.max_blocks

        self.collection.train.config.burnin_steps = self.train.actor_critic.burnin_steps
        self.collection.test.config.burnin_steps = self.train.actor_critic.burnin_steps

        self.train.actor_critic.imagine_horizon = self.common.seq_len

        self.eval.tokenizer.batch_num_samples = self.train.tokenizer.batch_num_samples
        self.eval.tokenizer.start_after_epochs = self.train.tokenizer.start_after_epochs

        self.eval.world_model.batch_num_samples = self.train.world_model.batch_num_samples
        self.eval.world_model.start_after_epochs = self.train.world_model.start_after_epochs

        self.eval.actor_critic.num_episodes_to_save = self.train.actor_critic.batch_num_samples
        self.eval.actor_critic.horizon = self.train.actor_critic.imagine_horizon
        self.eval.actor_critic.start_after_epochs = self.train.actor_critic.start_after_epochs
