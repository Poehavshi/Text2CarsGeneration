import logging
import os
from abc import ABC, abstractmethod
from typing import List
from torchvision.datasets.utils import download_url, download_and_extract_archive, download_file_from_google_drive, \
    extract_archive
from hydra import initialize, compose
import zipfile

log = logging.getLogger(__name__)


class ExtractOperator(ABC):
    """
    Perform some extract operations
    Factually this object is simple functor, which perform only one operation

    First type is download ready-to-use dataset from some source
    Second type is parser from some website
    """

    def __init__(self,
                 dataset_config,
                 output_directory: str):
        self.input_urls = dataset_config.urls
        self.output_dir = output_directory

    @abstractmethod
    def do(self):
        raise NotImplementedError("It's an abstract class for extract operator!")


def _is_archive(url):
    return url.endswith(".tgz") or url.endswith(".tar.gz")


class DownloadUrlOperator(ExtractOperator):
    def do(self):
        for url in self.input_urls:
            self.download_file(url)

    def download_file(self, url):
        if _is_archive(url):
            download_and_extract_archive(url, self.output_dir, remove_finished=True)
        else:
            download_url(url, self.output_dir)


class DownloadGoogleDriveOperator(ExtractOperator):
    def __init__(self, dataset_config, output_directory: str):
        super().__init__(dataset_config, output_directory)
        self.filenames = dataset_config.filenames

    def do(self):
        for url, filename in zip(self.input_urls, self.filenames):
            self.download_file(url, filename)

    def download_file(self, url, filename):
        root = self.output_dir
        download_file_from_google_drive(file_id=url, root=root, filename=filename)
        if filename.endswith(".rar"):
            extract_archive(from_path=os.path.join(root, filename), to_path=root, remove_finished=True)


# todo move to another file
EXTRACT_OPERATOR_INJECTOR = {
    "compcars": DownloadGoogleDriveOperator,
    "stanford": DownloadUrlOperator,
    "resized_DVM": DownloadUrlOperator,
    "carconnection": DownloadGoogleDriveOperator
}


if __name__ == '__main__':
    initialize(config_path=r"..\..\conf", job_name="carconnection_extract", version_base=None)
    cfg = compose(config_name="carconnection")
    root_path = cfg.raw_data_root
    for dataset in cfg.datasets:
        dataset_name = cfg.datasets[dataset].name
        dataset_path = os.path.join(root_path, dataset_name)

        extract_operator = EXTRACT_OPERATOR_INJECTOR[dataset_name](
            cfg.datasets[dataset],
            dataset_path
        )
        extract_operator.do()
