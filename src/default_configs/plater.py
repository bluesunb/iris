from dataclasses import dataclass
from typing import Callable, Dict, Any


class HydraWrapper:
    __target__: Callable = None

    def asdict(self, recursive: bool = True) -> Dict[str, Any]:
        if recursive:
            return {k: v.asdict() if isinstance(v, HydraWrapper) else v for k, v in self.__dict__.items() if k != '__target__'}
        else:
            return {k: v for k, v in self.__dict__.items() if k != '__target__'}
    

def instantiate(cfg: HydraWrapper):
    if not isinstance(cfg, (HydraWrapper, dict)):
        return cfg
    for k, v in cfg.asdict(recursive=False).items():
        if isinstance(v, HydraWrapper):
            cfg.__dict__[k] = instantiate(v)
    return cfg.__target__(**{k: instantiate(v) for k, v in cfg.asdict().items()})
