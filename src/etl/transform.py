"""
Module to transform images (resizing, filtering, synthesis images creation)
"""
import logging
import os
from abc import ABC, abstractmethod
from tqdm import tqdm

from hydra import initialize, compose
from scipy.io import loadmat
import pandas as pd

log = logging.getLogger(__name__)


class TransformOperator(ABC):
    """
    Operator for transform operations performed on the whole dataset
    """

    def __init__(self, input_dir, delete_old_meta=False):
        self.input_dir = input_dir
        self.delete_old_meta = delete_old_meta

    @abstractmethod
    def do(self):
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
            enumerated_classes.append({"class_index": class_index + 1, "class_name": class_name[0]})
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


class CompCarsTransform(TransformOperator):
    META_FILE = "data/misc/make_model_name.mat"

    def do(self):
        data_meta_df = self.transform_meta_for_data()
        sv_data_meta_df = self.transform_meta_for_sv_data()
        meta_df = pd.concat([data_meta_df, sv_data_meta_df])

        meta_df.to_csv(os.path.join(self.input_dir, "normalized_meta.csv"), index=False)

    def transform_meta_for_data(self):
        make_model_names = loadmat(os.path.join(self.input_dir, "data/misc/make_model_name.mat"))
        make_names_mapping = self._read_mapping_for_data(make_model_names['make_names'])
        model_names_mapping = self._read_mapping_for_data(make_model_names['model_names'])

        meta_df = self.create_meta_df_for_data(make_names_mapping, model_names_mapping)

        meta_df.to_csv(os.path.join(self.input_dir, "normalized_meta.csv"), index=False)
        return meta_df

    def _read_mapping_for_data(self, mat_array):
        mapping = {}
        for index, class_name in enumerate(mat_array):
            if len(class_name[0]):
                mapping[index + 1] = class_name[0][0]
            else:
                mapping[index + 1] = ''
        return mapping

    def create_meta_df_for_data(self, make_names_mapping, model_names_mapping) -> pd.DataFrame:
        # TODO add information about attributes (door number, seat_number and car type)
        # todo add information about viewpoint (from label directory)
        img_dir = os.path.join(self.input_dir, "data/image")
        df_dicts = []
        for root, subdirectories, files in os.walk(img_dir):
            for file in files:
                img_dir, make_id, model_id, year = root.split('\\')[-4:]
                img_dir = img_dir.replace('/', '\\')
                fpath = os.path.join(img_dir, make_id, model_id, year, file)

                brand = make_names_mapping[int(make_id)].strip()
                model = model_names_mapping[int(model_id)].replace(brand, '').strip()
                year = year.strip()

                caption = ' '.join([brand, model, year])

                df_dicts.append({'fpath': fpath, 'caption': caption})

        return pd.DataFrame(df_dicts)

    def create_meta_df_for_sv_data(self, model_names_mapping):
        # TODO read color list also
        # TODO add information about front view
        img_dir = os.path.join(self.input_dir, "sv_data/image")
        df_dicts = []
        for root, subdirectories, files in os.walk(img_dir):
            for file in files:
                img_dir, model_id = root.split('\\')[-2:]
                img_dir = img_dir.replace('/', '\\')

                fpath = os.path.join(img_dir, model_id, file)
                model = model_names_mapping[int(model_id)]
                df_dicts.append({'fpath': fpath, 'caption': model})
        return pd.DataFrame(df_dicts)

    def read_mapping_for_sv_data(self):
        make_model_names = loadmat(os.path.join(self.input_dir, "sv_data/sv_make_model_name.mat"))['sv_make_model_name']
        mapping = {}
        for index, make_model_name in enumerate(make_model_names):
            brand = make_model_name[0][0].strip()
            model_name = make_model_name[1][0].replace(brand, '').strip()
            class_index = index + 1
            mapping[class_index] = ' '.join([brand, model_name])
        return mapping

    def transform_meta_for_sv_data(self):
        mapping = self.read_mapping_for_sv_data()
        meta_df = self.create_meta_df_for_sv_data(mapping)
        return meta_df


class DVMCarTransform(TransformOperator):

    def do(self):
        meta_df = self.create_meta_df()
        print(meta_df)
        meta_df.to_csv(os.path.join(self.input_dir, "normalized_meta.csv"), index=False)

    def create_meta_df(self):
        df_dicts = []
        for root, subdirectories, files in tqdm(os.walk(self.input_dir), "Processing DVM file structure"):
            for file in files:
                if file != 'normalized_meta.csv':
                    brand, model, year, color = root.split('\\')[-4:]
                    caption = ' '.join([color, brand, model, year])
                    fpath = os.path.join(brand, model, year, color, file)
                    df_dicts.append({'fpath': fpath, 'caption': caption})
        return pd.DataFrame(df_dicts)


TRANSFORM_OPERATOR_INJECTOR = {
    "stanford": StanfordTransform,
    "compcars": CompCarsTransform,
    "resized_DVM": DVMCarTransform
}

if __name__ == '__main__':
    initialize(config_path=r"..\..\conf", job_name="dvmcars_extract", version_base=None)
    cfg = compose(config_name="dvmcar")
    root_input_dir = cfg.raw_data_root
    for dataset in cfg.datasets:
        dataset_name = cfg.datasets[dataset].name
        dataset_input_dir = os.path.join(root_input_dir, dataset_name)

        extract_operator = TRANSFORM_OPERATOR_INJECTOR[dataset_name](
            dataset_input_dir
        )
        extract_operator.do()
