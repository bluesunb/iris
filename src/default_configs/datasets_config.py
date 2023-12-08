from dataclasses import dataclass
from src.dataset import EpisodeDataset, EpisodeRamMonitoringDataset
from src.default_configs.plater import HydraWrapper


@dataclass
class DatasetTrainCfg(HydraWrapper):
    """
    See dataset.EpisodeDatasetRamMonitoring for more details.
    """
    __target__ = EpisodeRamMonitoringDataset
    max_ram_usage: str = "30G"
    name: str = "train_dataset"


@dataclass
class DatasetTestCfg(HydraWrapper):
    """
    See dataset.EpisodeDataset for more details.
    """
    __target__ = EpisodeDataset
    max_num_episodes: int = None
    name: str = "test_dataset"


@dataclass
class DatasetCfg(HydraWrapper):
    train: DatasetTrainCfg = DatasetTrainCfg()
    test: DatasetTestCfg = DatasetTestCfg()


if __name__ == "__main__":
    from src.default_configs.plater import instantiate
    dataset_cfg = DatasetTrainCfg()
    instantiate(dataset_cfg)
