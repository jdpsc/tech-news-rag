from beam import Image, Volume, function

from .train_run import train


@function(
    gpu=["T4", "A10G"],
    cpu=4,
    memory="32Gi",
    image=Image(python_version="python3.10", python_packages="requirements.txt"),
    volumes=[
        Volume(mount_path="./dataset", name="dataset"),
        Volume(mount_path="./output", name="train_qa_output"),
        Volume(mount_path="./model_cache", name="model_cache"),
        Volume(mount_path="./logs", name="logs"),
    ],
    keep_warm_seconds=300,
)
def beam_train():

    config_file = "configs/training_config.yaml"
    output_dir = "./output"
    dataset_dir = "./dataset"
    env_file_path = ".env"
    logging_config_path = "logging.yaml"
    model_cache_dir = "./model_cache"

    train(
        config_file,
        output_dir,
        dataset_dir,
        env_file_path,
        logging_config_path,
        model_cache_dir,
    )


if __name__ == "__main__":
    beam_train.remote()
