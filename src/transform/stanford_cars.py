import os.path

import luigi
import pandas as pd

from src.dependencies.extract_dag import extract_tasks


class StanfordCarsProcessing(luigi.Task):
    annotations_path: str = luigi.Parameter()
    output_annotations_path: str = luigi.Parameter()

    def requires(self):
        return extract_tasks["stanford_cars"]

    def run(self):
        self.output().makedirs()
        self.transform_meta()

    def output(self):
        return luigi.LocalTarget(self.output_annotations_path)

    def transform_meta(self):
        meta_df = self.construct_meta_information(os.path.join(self.annotations_path, "test"))
        meta_df_2 = self.construct_meta_information(os.path.join(self.annotations_path, "train"))
        meta_df = pd.concat([meta_df, meta_df_2])
        meta_df.to_csv(self.output().path, index=False)

    def construct_meta_information(self, dataset_path):
        """
        Construct meta information from the dirs, which have names of classes
        """
        meta = []
        for class_dir in os.listdir(dataset_path):
            class_dir_path = os.path.join(dataset_path, class_dir)
            for file in os.listdir(class_dir_path):
                file_path = os.path.join(class_dir_path, file)
                meta.append([file_path, class_dir])
        return pd.DataFrame(meta, columns=["path", "class"])
