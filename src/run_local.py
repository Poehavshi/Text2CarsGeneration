import luigi

from src.dependencies.transform_dag import tasks


def run() -> None:
    luigi.build(tasks, local_scheduler=True)


if __name__ == "__main__":
    run()
