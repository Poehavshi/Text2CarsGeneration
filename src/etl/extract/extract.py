"""
Classes to
1) Download files from some url (source .py file)
2) Extract it from archive to cache folder (decompressor .py file)
3) Creation of unified .csv files of every image (this .py file) (todo maybe .feather or .pickle?)
"""

from abc import ABC, abstractmethod


class AbstractExtractor(ABC):
    def __init__(self, download_urls: list[str], folder: str):
        self.urls = download_urls
        self.folder = folder

    @abstractmethod
    def run(self):
        """
        Run extract pipeline:
        1) Download
        2) Decompress
        3) Load all meta information
        """
        raise NotImplementedError("It's base extractor!")


class StanfordDatasetExtractor(AbstractExtractor):
    def run(self):
        pass
