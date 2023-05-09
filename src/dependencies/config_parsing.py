import logging
import os.path
from dataclasses import dataclass, field

from hydra import compose, initialize

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    urls: list
    google_drive_ids: dict


@dataclass
class MetaInfo:
    annotations_path: str
    output_annotations_path: str


@dataclass
class Config:
    root_dir: str = "../../"
    raw_data_path: str = os.path.join(root_dir, "data/raw")
    meta_data_path: str = os.path.join(root_dir, "data/meta")
    datasets: dict[Dataset] = field(default_factory=dict)
    meta_infos: dict[MetaInfo] = field(default_factory=dict)


# context initialization
with initialize(version_base=None, config_path="../../config", job_name="test_app"):
    # compose the config
    config = compose(config_name="config")
    print(os.path.abspath(config.raw_data_path))
    config = Config(**config)
