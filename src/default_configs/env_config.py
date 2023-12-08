from dataclasses import dataclass
from src.envs.wrappers import make_atari
from src.default_configs.plater import HydraWrapper


@dataclass
class EnvTrainCfg(HydraWrapper):
    """
    See envs.wrappers.make_atari for more details.
    """
    __target__ = lambda self, **kwargs: make_atari(**kwargs)
    env_id: str = "BreakoutNoFrameskip-v4"
    size: int = 64
    max_episode_steps: int = 20000
    noop_max: int = 30
    frame_skip: int = 4
    done_on_life_loss: bool = True
    clip_reward: bool = False

    def __post_init__(self):
        # self.__target__ = make_atari
        if self.env_id is None:
            raise ValueError('env_id must be specified')


@dataclass
class EnvTestCfg(HydraWrapper):
    """
    See envs.wrappers.make_atari for more details.
    """
    __target__ = lambda self, **kwargs: make_atari(**kwargs)
    env_id: str = EnvTrainCfg.env_id
    size: int = EnvTrainCfg.size
    max_episode_steps: int = 108000
    noop_max: int = 1
    frame_skip: int = EnvTrainCfg.frame_skip
    done_on_life_loss: bool = False
    clip_reward: bool = False


@dataclass
class EnvCfg(HydraWrapper):
    train: EnvTrainCfg = EnvTrainCfg()
    test: EnvTestCfg = EnvTestCfg()
    keymap: str = 'atari/' + train.env_id

    def __post_init__(self):
        self.test.env_id = self.train.env_id
        self.test.size = self.train.size
        self.test.frame_skip = self.train.frame_skip
        self.keymap = 'atari/' + self.train.env_id


if __name__ == "__main__":
    from src.default_configs.plater import instantiate
    env_cfg = EnvTrainCfg()
    instantiate(env_cfg)