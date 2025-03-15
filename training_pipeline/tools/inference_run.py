from pathlib import Path

import fire

from training_pipeline import configs


def infer(
    config_file: str,
    dataset_dir: str,
    model_cache_dir: str,
    output_dir: str = "output-inference",
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
):
    """
    Run inference on a dataset using a trained model.

    Args:
        config_file (str): Path to the inference configuration file.
        dataset_dir (str): Path to the root directory of the dataset.
        output_dir (str, optional): Path to the output directory. Defaults to "output-inference".
        env_file_path (str, optional): Path to the environment variables file. Defaults to ".env".
        logging_config_path (str, optional): Path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str): Path to the directory where the trained model is cached.
    """

    import logging

    from training_pipeline import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from training_pipeline import utils
    from training_pipeline.api import InferenceAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file_path = Path(config_file)
    root_dataset_dir_path = Path(dataset_dir)
    model_cache_dir_path = Path(model_cache_dir)
    inference_output_dir_path = Path(output_dir)
    inference_output_dir_path.mkdir(exist_ok=True, parents=True)

    inference_config = configs.InferenceConfig.from_yaml(config_file_path)
    inference_api = InferenceAPI.from_config(
        config=inference_config,
        root_dataset_dir=root_dataset_dir_path,
        model_cache_dir=model_cache_dir_path,
    )
    inference_api.infer_all(
        output_file=inference_output_dir_path / "output-inference-api.json"
    )


if __name__ == "__main__":
    fire.Fire(infer)
