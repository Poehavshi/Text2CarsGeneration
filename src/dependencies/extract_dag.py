from src.dependencies.config_parsing import config
from src.extract.extract import ExtractRawDataset

print(config.raw_data_path)

extract_tasks = {
    dataset_name: ExtractRawDataset(
        config.raw_data_path, dataset_name, list(dataset.urls), dict(dataset.google_drive_ids)
    )
    for dataset_name, dataset in config.datasets.items()
}
