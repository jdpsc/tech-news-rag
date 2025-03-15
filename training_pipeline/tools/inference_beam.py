from beam import Image, Volume, function

from .inference_run import infer


@function(
    gpu=["T4", "A10G"],
    cpu=4,
    memory="32Gi",
    image=Image(python_version="python3.10", python_packages="requirements.txt"),
    volumes=[
        Volume(mount_path="./dataset", name="dataset"),
        Volume(mount_path="./output-inference", name="testing_qa_output"),
        Volume(mount_path="./model_cache", name="model_cache"),
        Volume(mount_path="./logs", name="logs"),
    ],
)
def infer_beam():

    config_file = "configs/inference_config.yaml"
    dataset_dir = "./dataset"
    output_dir = "./output-inference"
    env_file_path = ".env"
    logging_config_path = "logging.yaml"
    model_cache_dir = "./model_cache"

    infer(
        config_file,
        dataset_dir,
        output_dir,
        env_file_path,
        logging_config_path,
        model_cache_dir,
    )


if __name__ == "__main__":
    infer_beam.remote()
