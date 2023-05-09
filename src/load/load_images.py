import os
from os.path import abspath

import luigi
import pandas as pd

from src.dependencies.transform_dag import transform_tasks


class FilterImages(luigi.Task):
    """
    Copy and filter images from input_dir to output_dir
    """
    meta_info_path: str = luigi.Parameter()
    output_dir: str = luigi.Parameter()

    def requires(self):
        return transform_tasks

    def run(self):
        self.output().makedirs()
        meta_info = pd.read_csv(self.meta_info_path)
        self.filter_images(meta_info)

    def filter_images(self, meta_info):
        """
        Copy only images that appear in meta_info
        """
        new_meta_info = []
        for index, row in meta_info.iterrows():
            img_path = row["path"]
            new_path = self.copy_image("../data/"+img_path, row["class"])
            new_meta_info.append({"path": new_path, "class": row["class"]})
        new_meta_info = pd.DataFrame(new_meta_info)
        new_meta_info.to_csv(os.path.join(self.output().path, "meta.csv"), index=False)

    def copy_image(self, img_path, class_name):
        """
        Copy image to output_dir/class_name
        """
        output_path = os.path.join(self.output_dir, class_name)
        output_path = output_path.replace(" ", "_")
        img_path = os.path.abspath(img_path)
        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)
        os.system(f"cp {img_path} {output_path}")
        return os.path.join(output_path, os.path.basename(img_path))

    def output(self):
        return luigi.LocalTarget(self.output_dir)
