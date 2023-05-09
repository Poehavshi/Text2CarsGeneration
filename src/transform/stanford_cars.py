import os.path

import luigi
import pandas as pd
from scipy.io.matlab import loadmat

from src.dependencies.extract_dag import extract_tasks


class StanfordCarsProcessing(luigi.Task):
    annotations_path: str = luigi.Parameter()
    output_annotations_path: str = luigi.Parameter()

    def run(self):
        self.output().makedirs()
        self.transform_meta()

    def output(self):
        return luigi.LocalTarget(self.output_annotations_path)

    def transform_meta(self):
        meta_df = self.read_meta_information()
        meta_df.to_csv(self.output().path, index=False)

    def read_meta_information(self):
        print(os.path.abspath(self.annotations_path))
        meta_array = loadmat(self.annotations_path)
        print(meta_array)
        annotations = self._parse_annotations(meta_array["annotations"])
        class_names = self._parse_class_names(meta_array["class_names"])
        meta_df = pd.merge(annotations, class_names, on="class_index")
        meta_df.drop("class_index", inplace=True, axis=1)
        meta_df.rename(columns={"class_name": "caption"}, inplace=True)
        return meta_df

    @staticmethod
    def _parse_class_names(class_names):
        class_names = class_names.flatten()
        enumerated_classes = []
        for class_index, class_name in enumerate(class_names):
            enumerated_classes.append({"class_index": class_index + 1, "class_name": class_name[0]})
        return pd.DataFrame(enumerated_classes)

    @staticmethod
    def _parse_annotations(annotations):
        annotations = annotations.flatten()
        annotations_dicts = []
        for annotation in annotations:
            fpath = str(annotation[0][0])
            class_index = int(annotation[5][0][0])
            annotations_dicts.append({"class_index": class_index, "fpath": fpath})
        return pd.DataFrame(annotations_dicts)
