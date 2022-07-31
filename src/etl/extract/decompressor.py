"""
Classes to decompress files to cache folder
perform ordering of all files - separate meta and image files
"""

from abc import ABC, abstractmethod
import os
import tarfile


class AbstractDecompressor(ABC):
    def __init__(self, name: str, config):
        self.folder_with_data = os.path.join(config.cache_folder, name)
        self.folder_to_save_images = os.path.join(self.folder_with_data, 'images')
        self.meta_file_path = os.path.join(self.folder_with_data, 'meta.csv')

    @abstractmethod
    def decompress(self):
        """
        Parse files in folder, extract all archives
        """
        raise NotImplementedError("It's base class of decompressor!")

    @abstractmethod
    def order(self):
        """
        Find all image files and move it to separate folder
        """
        raise NotImplementedError("It's base class of decompressor!")


class TarDecompressor(AbstractDecompressor):
    def order(self):
        pass

    def decompress(self):
        for path, directories, files in os.walk(self.folder_with_data):
            for f in files:
                if f.endswith(".tar.gz"):
                    tar = tarfile.open(os.path.join(path, f), 'r:gz')
                    tar.extractall(path=path)
                    tar.close()
