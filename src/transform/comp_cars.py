import os.path

import luigi
import pandas as pd
from scipy.io import loadmat

from src.dependencies.extract_dag import extract_tasks


class CompCarsProcessing(luigi.Task):
    annotations_path: str = luigi.Parameter()
    output_annotations_path: str = luigi.Parameter()

    def requires(self):
        return extract_tasks["comp_cars"]

    def run(self):
        self.output().makedirs()
        meta_df = self.transform_meta()
        meta_df.to_csv(self.output().path, index=False)

    def transform_meta(self):
        make_model_names = loadmat(os.path.join(self.annotations_path, "misc", "make_model_name.mat"))
        make_names_mapping = self._read_mapping_for_data(make_model_names["make_names"])
        model_names_mapping = self._read_mapping_for_data(make_model_names["model_names"])
        meta_df = self.create_meta_df_for_data(make_names_mapping, model_names_mapping)
        return meta_df

    def _read_mapping_for_data(self, mat_array):
        mapping = {}
        for index, class_name in enumerate(mat_array):
            if len(class_name[0]):
                mapping[index + 1] = class_name[0][0]
            else:
                mapping[index + 1] = ""
        return mapping

    def create_meta_df_for_data(self, make_names_mapping, model_names_mapping) -> pd.DataFrame:
        img_dir = os.path.join(self.annotations_path, "image")
        df_dicts = []
        for root, _subdirectories, files in os.walk(img_dir):
            for file in files:
                img_dir, make_id, model_id, year = root.split("/")[-4:]
                fpath = os.path.join(img_dir, make_id, model_id, year, file)
                brand = make_names_mapping[int(make_id)].strip()
                model = model_names_mapping[int(model_id)].replace(brand, "").strip()
                year = year.strip()
                caption = " ".join([brand, model, year])
                df_dicts.append({"path": os.path.join(self.annotations_path, fpath), "class": caption})
        return pd.DataFrame(df_dicts)

    def output(self):
        return luigi.LocalTarget(self.output_annotations_path)
