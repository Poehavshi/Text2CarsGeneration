"""
Classes, which using strategy pattern to download images from some sources in cache
"""

from abc import ABC, abstractmethod

from hydra import initialize, compose
from torchvision.datasets.utils import download_url
import os


class AbstractSource(ABC):
    def __init__(self, name: str, config):
        self.folder_to_save = os.path.join(config.cache_folder, name)

    @abstractmethod
    def download(self):
        """
        Download all files to cache folder
        """
        raise NotImplementedError("It's base class of dataset!")


class WebDatasetSource(AbstractSource):
    def __init__(self, urls: list[str], name: str, config):
        super().__init__(name, config)
        self.urls = urls

    def download(self):
        for url in self.urls:
            download_url(url, self.folder_to_save)


if __name__ == '__main__':
    initialize(config_path="..\..\..\conf", job_name="test_app", version_base=None)
    cfg = compose(config_name="config")
    test_source = WebDatasetSource(
        ["http://ai.stanford.edu/~jkrause/car196/car_ims.tgz", "http://ai.stanford.edu/~jkrause/car196/cars_annos.mat"],
        "stanford",
        cfg
    )
    test_source.download()
