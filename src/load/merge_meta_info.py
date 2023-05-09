import os

import luigi
import pandas as pd

from src.dependencies.transform_dag import transform_tasks


class MergeMetaInformation(luigi.Task):
    input_dir: str = luigi.Parameter()
    output_file: str = luigi.Parameter()

    def requires(self):
        return transform_tasks

    def run(self):
        self.output().makedirs()
        meta_info = self.merge_meta_information()
        # sort by path
        meta_info = meta_info.sort_values(by=["path"], ignore_index=True, key=lambda x: x.str.lower(), ascending=False)
        meta_info = self.remove_year_from_class(meta_info)
        meta_info = self.leave_only_k_for_each_class(meta_info)
        meta_info["path"] = meta_info["path"].str.replace("../data/", "")
        print(len(meta_info))
        meta_info.to_csv(self.output().path, index=False)

    def merge_meta_information(self):
        meta = []
        for task in os.listdir(self.input_dir):
            df = pd.read_csv(os.path.join(self.input_dir, task))
            meta.append(df)
        return pd.concat(meta)

    def leave_only_k_for_each_class(self, meta_info, k=5):
        meta_info = meta_info.groupby("class").head(k)
        return meta_info

    def remove_year_from_class(self, meta_info):
        meta_info["class"] = meta_info["class"].str[:-5]
        return meta_info

    def output(self):
        return luigi.LocalTarget(self.output_file)
