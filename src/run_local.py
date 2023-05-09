import luigi

from src.dependencies.load_dag import load_tasks


def run() -> None:
    luigi.build(load_tasks)


if __name__ == "__main__":
    run()
