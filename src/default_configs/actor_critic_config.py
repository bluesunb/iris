from dataclasses import dataclass
from src.default_configs.plater import HydraWrapper


@dataclass
class ActorCriticCfg(HydraWrapper):
    use_real: bool = False
