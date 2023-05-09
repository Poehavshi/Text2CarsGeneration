from src.extract.extract import ExtractRawDataset
from src.dependencies.config_parsing import config

print(config.raw_data_path)

extract_tasks: list[ExtractRawDataset] = [
    ExtractRawDataset(config.raw_data_path, dataset_name, list(dataset.urls), dict(dataset.google_drive_ids))
    for dataset_name, dataset in config.datasets.items()
]

processing_tasks = []

tasks = extract_tasks + processing_tasks
