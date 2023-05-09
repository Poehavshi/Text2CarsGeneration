"""
Steps to download all necessary files
"""

import os

import gdown
import luigi
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    extract_archive,
)
from tqdm import tqdm


class ExtractRawDataset(luigi.Task):
    output_dir: str = luigi.Parameter()
    dataset_name: str = luigi.Parameter()
    urls: str = luigi.ListParameter()
    google_drive_ids: dict = luigi.DictParameter()

    def run(self):
        self.output().makedirs()
        for url in tqdm(self.urls, desc="Downloading files from common urls"):
            self.download_file(url)
        for drive_id, filename in tqdm(self.google_drive_ids.items(), desc="Downloading files from Google drive"):
            self.download_google_drive_file(drive_id, filename)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_dir, self.dataset_name))

    def download_file(self, url):
        if _is_archive(url):
            download_and_extract_archive(url, self.output().path, remove_finished=True)
        else:
            download_url(url, self.output().path)

    def download_google_drive_file(self, google_drive_id, filename):
        root = self.output().path
        os.makedirs(root, exist_ok=True)
        gdown.download(url=google_drive_id, output=os.path.join(root, filename), quiet=False)
        if _is_archive(filename):
            extract_archive(from_path=os.path.join(root, filename), to_path=root, remove_finished=True)


def _is_archive(url):
    return url.endswith(".tgz") or url.endswith(".tar.gz") or url.endswith(".rar")
