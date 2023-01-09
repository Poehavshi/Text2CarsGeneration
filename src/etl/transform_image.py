import shutil
from pathlib import Path

import pandas as pd
import os
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision
import torchvision.transforms.functional as F


class FilterOperator:
    """
    Delete files, that not appear in meta information file
    """
    def __init__(self, input_dir, sub_dir = None):
        self.input_dir = input_dir
        self.sub_dir = sub_dir

    def do(self):
        meta_info_csv = os.path.join(self.input_dir, "normalized_meta.csv")
        meta_info = pd.read_csv(meta_info_csv)

        img_dir = os.path.join(self.input_dir, self.sub_dir)
        for root, subdirectories, files in tqdm(os.walk(img_dir), "Processing dataset file structure"):
            for file in files:
                fpath = os.path.join(self.sub_dir, file).replace('\\', '/')
                if fpath not in meta_info['fpath'].unique():
                    os.remove(os.path.join(root, file))


class MergeMetaInformationOperator:
    """
    Find meta information files and merge it
    """
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def do(self):
        meta_infos = [self.open_meta_information(meta_info_path) for meta_info_path in self.find_all_meta()]
        df = pd.concat(meta_infos)
        print(df)
        df.to_csv(os.path.join(self.input_dir, "meta_info.csv"), index=False)

    def open_meta_information(self, meta_info_path):
        df = pd.read_csv(os.path.join(self.input_dir, meta_info_path))
        dataset_name = meta_info_path.split('\\')[0]
        df['fpath'] = dataset_name + '\\' + df['fpath'].astype(str)
        return df

    def find_all_meta(self):
        files = [os.path.join(directory, "normalized_meta.csv") for directory in os.listdir(self.input_dir) if directory!="meta_info.csv" and directory != "datasets.zip"]
        return files



class ResizeOperator:
    """
    Resize all images to the same size
    """
    def __init__(self, input_dir, image_size = 256):
        self.input_dir = input_dir
        self.image_size = image_size

    def do(self):
        resize_transform = torchvision.transforms.Resize((self.image_size, self.image_size))
        for file in tqdm(os.listdir(r"E:\University\НИР\Project\src\etl\datasets\stanford\car_ims"), "Resizing dataset "):
            file_path = os.path.join(r"E:\University\НИР\Project\src\etl\datasets\stanford\car_ims", file)
            image = read_image(file_path).to("cuda:0")
            new_image = F.to_pil_image(resize_transform(image))
            new_image.save(file_path)


class CreateSubset:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def do(self):
        meta_info = pd.read_csv(os.path.join(self.input_dir, "meta_info.csv"))
        bmw_subset = meta_info[(meta_info['caption'].str.contains("BWM")) | (meta_info['caption'].str.contains("BMW"))]
        os.makedirs(self.output_dir, exist_ok=True)

        for index, bmw_row in bmw_subset.iterrows():
            input_path = os.path.join(self.input_dir, bmw_row['fpath'])
            output_path = os.path.join(self.output_dir, bmw_row['fpath'])
            os.makedirs(Path(output_path).parent, exist_ok=True)
            shutil.copy(input_path, output_path)

        bmw_subset.to_csv(os.path.join(self.output_dir, "meta_info.csv"))



if __name__ == '__main__':
    # FilterOperator(r"E:\University\НИР\Project\src\etl\datasets\carconnection", "data").do()
    # MergeMetaInformationOperator(r"E:\University\НИР\Project\src\etl\datasets").do()
    # ResizeOperator(r"E:\University\НИР\Project\src\etl\datasets").do()
    CreateSubset(r"E:\University\НИР\Project\src\etl\datasets", r"E:\University\НИР\Project\src\etl\datasets\subset").do()

