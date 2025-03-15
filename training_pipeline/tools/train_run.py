from pathlib import Path

import fire

from training_pipeline import configs


def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    model_cache_dir: str,
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
):
    """
    Trains a machine learning model using the specified configuration file and dataset.

    Args:
        config_file (str): The path to the configuration file for the training process.
        output_dir (str): The directory where the trained model will be saved.
        dataset_dir (str): The directory where the training dataset is located.
        env_file_path (str, optional): The path to the environment variables file. Defaults to ".env".
        logging_config_path (str, optional): The path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str): The directory where the trained model will be cached.
    """

    import logging

    from training_pipeline import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from training_pipeline import utils
    from training_pipeline.api import TrainingAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file_path = Path(config_file)
    output_dir_path = Path(output_dir)
    root_dataset_dir_path = Path(dataset_dir)
    model_cache_dir_path = Path(model_cache_dir)

    training_config = configs.TrainingConfig.from_yaml(
        config_file_path, output_dir_path
    )
    training_api = TrainingAPI.from_config(
        config=training_config,
        root_dataset_dir=root_dataset_dir_path,
        model_cache_dir=model_cache_dir_path,
    )
    training_api.train()


if __name__ == "__main__":
    fire.Fire(train)
