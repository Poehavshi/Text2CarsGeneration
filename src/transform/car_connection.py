import os.path

import luigi
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from src.dependencies.extract_dag import extract_tasks

RESNET_CAR_IDX = {656, 627, 817, 511, 468, 751, 705, 757, 717, 734, 654, 675, 864, 609, 436}


class CarConnectionProcessing(luigi.Task):
    annotations_path: str = luigi.Parameter()
    output_annotations_path: str = luigi.Parameter()

    def requires(self):
        return extract_tasks["car_connection"]

    def run(self):
        self.output().makedirs()
        weights = ResNet50_Weights.DEFAULT
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = resnet50(weights=weights)
        model.eval().to(device)
        preprocess = weights.transforms()
        meta_df = self.create_meta_df(preprocess, model, device)
        print(meta_df)
        meta_df.to_csv(self.output().path, index=False)

    def is_car_image(self, image_path, preprocess, model, device):
        return True
        img = read_image(image_path)
        batch = preprocess(img).unsqueeze(0).to(device)
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        proba = prediction[class_id].item()
        return class_id in RESNET_CAR_IDX and proba > 0.2

    def create_meta_df(self, preprocess, classifier, device):
        df_dicts = []
        img_dir = self.annotations_path
        for _root, _subdirectories, files in tqdm(os.walk(img_dir), "Processing CarConnection file structure"):
            for file in files:
                try:
                    brand, model, year = file.split("_")[:3]
                except ValueError:
                    print(file)
                    continue
                caption = " ".join([brand, model, year])
                fpath = os.path.join(img_dir, file)
                if self.is_car_image(fpath, preprocess, classifier, device):
                    df_dicts.append({"path": fpath, "class": caption})
        return pd.DataFrame(df_dicts)

    def output(self):
        return luigi.LocalTarget(self.output_annotations_path)
