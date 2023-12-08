import hydra
from omegaconf import DictConfig
from src.trainer import Trainer
from src.config import Config


# @hydra.main(config_path="../config", config_name="config")
# def main(config: DictConfig):
def main(config: Config):
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    config = Config()
    
    config.collection.test.num_envs = 2
    config.tokenizer.emb_dim = 360
    
    main(config)
