from src.dependencies.config_parsing import MetaInfo, config
from src.transform.comp_cars import CompCarsProcessing
from src.transform.stanford_cars import StanfordCarsProcessing

stanford_meta_info: MetaInfo = config.meta_infos["stanford_cars"]
comp_cars_meta_info: MetaInfo = config.meta_infos["comp_cars"]
processing_tasks = {
    "stanford_cars": StanfordCarsProcessing(
        stanford_meta_info.annotations_path, stanford_meta_info.output_annotations_path
    ),
    "comp_cars": CompCarsProcessing(
        comp_cars_meta_info.annotations_path, comp_cars_meta_info.output_annotations_path
    )
}

tasks = processing_tasks.values()
