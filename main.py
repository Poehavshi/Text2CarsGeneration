import hydra
from omegaconf import DictConfig, OmegaConf

import os


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print(os.getcwd())


if __name__ == '__main__':
    main()
