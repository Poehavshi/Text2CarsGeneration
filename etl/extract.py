from abc import ABC, abstractmethod
import tarfile

from torchvision.datasets.utils import download_url

# download_url('http://something.com/file.zip', '~/my_folder`)

# fixme move it to some config file
CACHE_FOLDER = '../tmp'


class DatasetSource(ABC):
    def __init__(self, download_urls: list[str], folder: str):
        self.urls = download_urls
        self.folder = folder

    @abstractmethod
    def download(self):
        raise NotImplementedError("It's base class of dataset source!")

    @abstractmethod
    def extract(self):
        raise NotImplementedError("It's base class of dataset source!")


class StanfordDatasetSource(DatasetSource):
    def __init__(self, download_urls: list[str], folder: str):
        super().__init__(download_urls, folder)

    def download(self):
        for url in self.urls:
            download_url(url, CACHE_FOLDER)

    def extract(self):
        pass


if __name__ == '__main__':
    dataset = StanfordDatasetSource(['http://ai.stanford.edu/~jkrause/car196/cars_annos.mat'], '')
    dataset.download()
