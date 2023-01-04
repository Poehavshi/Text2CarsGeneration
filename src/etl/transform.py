"""
Module to transform images (resizing, filtering, synthesis images creation)
"""
import os
from abc import ABC, abstractmethod

from hydra import initialize, compose
from scipy.io import loadmat
import pandas as pd


class TransformOperator(ABC):
    """
    Operator for transform operations performed on the whole dataset
    """
    def __init__(self, input_dir, output_dir, delete_old_meta=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.delete_old_meta = delete_old_meta

    @abstractmethod
    def do(self):
        raise NotImplementedError("It's an abstract class for transform operator!")

    @abstractmethod
    def transform_meta(self):
        # Dataframe with meta information always have structure like this:
        #
        # caption        | path
        # some_caption1  | some_path_to_image1
        # some_caption2  | some_path_to_image2
        raise NotImplementedError("It's an abstract class for transform operator!")


class StanfordTransform(TransformOperator):
    META_FILE = "cars_annos.mat"

    def do(self):
        self.transform_meta()

    def transform_meta(self):
        meta_df = self.read_meta_information()
        meta_df.drop('class_index', inplace=True, axis=1)
        meta_df.rename(columns={"class_name": "caption"}, inplace=True)
        meta_df.to_csv(os.path.join(self.input_dir, "normalized_meta.csv"), index=False)

        if self.delete_old_meta:
            old_meta_file = os.path.join(self.input_dir, StanfordTransform.META_FILE)
            os.remove(old_meta_file)

    def read_meta_information(self):
        meta_array = loadmat(os.path.join(self.input_dir, StanfordTransform.META_FILE))
        annotations = self._parse_annotations(meta_array['annotations'])
        class_names = self._parse_class_names(meta_array['class_names'])

        meta_df = pd.merge(annotations, class_names, on="class_index")
        return meta_df

    @staticmethod
    def _parse_class_names(class_names):
        class_names = class_names.flatten()
        enumerated_classes = []
        for class_index, class_name in enumerate(class_names):
            enumerated_classes.append({"class_index": class_index+1, "class_name": class_name[0]})
        return pd.DataFrame(enumerated_classes)

    @staticmethod
    def _parse_annotations(annotations):
        annotations = annotations.flatten()
        annotations_dicts = []
        for annotation in annotations:
            fpath = str(annotation[0][0])
            class_index = int(annotation[5][0][0])
            annotations_dicts.append({"class_index": class_index,
                                      "fpath": fpath})
        return pd.DataFrame(annotations_dicts)


if __name__ == '__main__':
    initialize(config_path=r"..\..\conf", job_name="stanford_extract", version_base=None)
    cfg = compose(config_name="stanford")
    root_input_dir = cfg.raw_data_root
    root_output_dir = cfg.transformed_data_root
    for dataset in cfg.datasets:
        dataset_name = cfg.datasets[dataset].name
        dataset_input_dir = os.path.join(root_input_dir, dataset_name)
        dataset_output_dir = os.path.join(root_output_dir, dataset_name)

        extract_operator = StanfordTransform(
            dataset_input_dir,
            dataset_output_dir
        )
        extract_operator.do()
