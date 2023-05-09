"""
Module to transform images (resizing, filtering, synthesis images creation)
"""
import logging
import os
from abc import ABC, abstractmethod

import pandas as pd
import torch
from hydra import compose, initialize
from scipy.io import loadmat
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

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


class CarConnectionTransform(TransformOperator):
    RESNET_CAR_IDX = {656, 627, 817, 511, 468, 751, 705, 757, 717, 734, 654, 675, 864, 609, 436}

    def __init__(self, input_dir, delete_old_meta=False):
        super().__init__(input_dir, delete_old_meta)
        # Step 1: Initialize model with the best available weights
        weights = ResNet50_Weights.DEFAULT
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=weights)
        self.model.eval().to(self.device)

        # Step 2: Initialize the inference transforms
        self.preprocess = weights.transforms()

    def is_car_image(self, image_path):
        img = read_image(image_path)
        # Step 3: Apply inference preprocessing transforms
        batch = self.preprocess(img).unsqueeze(0).to(self.device)

        # Step 4: Use the model and print the predicted category
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()

        return class_id in CarConnectionTransform.RESNET_CAR_IDX

    def do(self):
        meta_df = self.create_meta_df()
        print(meta_df)
        meta_df.to_csv(os.path.join(self.input_dir, "normalized_meta.csv"), index=False)

    def create_meta_df(self):
        df_dicts = []
        img_dir = os.path.join(self.input_dir, "data")
        for _root, _subdirectories, files in tqdm(os.walk(img_dir), "Processing CarConnection file structure"):
            for file in files:
                brand, model, year = file.split("_")[:3]
                caption = " ".join([brand, model, year])
                fpath = os.path.join("data", file)
                # if self.is_car_image(os.path.join(self.input_dir, fpath)):
                df_dicts.append({"fpath": fpath, "caption": caption})
        return pd.DataFrame(df_dicts)


TRANSFORM_OPERATOR_INJECTOR = {
    "carconnection": CarConnectionTransform,
}

if __name__ == "__main__":
    initialize(config_path=r"..\..\conf", job_name="carconnection_transform", version_base=None)
    cfg = compose(config_name="carconnection")
    root_input_dir = cfg.raw_data_root
    for dataset in cfg.datasets:
        dataset_name = cfg.datasets[dataset].name
        dataset_input_dir = os.path.join(root_input_dir, dataset_name)

        transform_operator = TRANSFORM_OPERATOR_INJECTOR[dataset_name](dataset_input_dir)
        transform_operator.do()
