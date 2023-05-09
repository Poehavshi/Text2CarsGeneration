import os

from src.dependencies.config_parsing import config
from src.load.load_images import FilterImages
from src.load.merge_meta_info import MergeMetaInformation

load_tasks = [MergeMetaInformation(config.meta_data_path, os.path.join(config.meta_data_path, "meta_all.csv")),
              FilterImages(os.path.join(config.meta_data_path, "meta_all.csv"), os.path.join(config.root_dir, "data", "final"))
              ]
