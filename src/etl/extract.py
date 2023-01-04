import logging
import os
from abc import ABC, abstractmethod
from typing import List
from torchvision.datasets.utils import download_url, download_and_extract_archive
import glob
from hydra import initialize, compose

log = logging.getLogger(__name__)

class ExtractOperator(ABC):
    """
    Perform some extract operations
    Factually this object is simple functor, which perform only one operation

    First type is download ready-to-use dataset from some source
    Second type is parser from some website
    """
    def __init__(self,
                 urls: List[str],
                 output_directory: str):
        self.input_urls = urls
        self.output_dir = output_directory

    @abstractmethod
    def do(self):
        raise NotImplementedError("It's an abstract class for extract operator!")


def _is_archive(url):
    return url.endswith(".tgz") or url.endswith(".tar.gz")


class DownloadOperator(ExtractOperator):
    def do(self):
        for url in self.input_urls:
            self.download_file(url)

    def download_file(self, url):
        if _is_archive(url):
            self.download_archive(url)
        else:
            download_url(url, self.output_dir)
    def download_archive(self, url):
        download_and_extract_archive(url, self.output_dir)
        for file in glob.glob(os.path.join(self.output_dir, "*.tgz")):
            log.info(f"Delete archive with name: {file}")
            os.remove(file)


if __name__ == '__main__':
    initialize(config_path=r"..\..\conf", job_name="stanford_extract", version_base=None)
    cfg = compose(config_name="stanford")
    root_path = cfg.raw_data_root
    for dataset in cfg.datasets:
        dataset_name = cfg.datasets[dataset].name
        dataset_path = os.path.join(root_path, dataset_name)

        extract_operator = DownloadOperator(
            cfg.datasets[dataset].urls,
            dataset_path
        )
        extract_operator.do()
