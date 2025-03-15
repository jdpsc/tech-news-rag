import logging
from pathlib import Path
from typing import Optional

import fire
from beam import Image, Volume, endpoint

logger = logging.getLogger(__name__)


# === Bot Loaders ===


def load_bot(
    model_cache_dir: Optional[str],
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    embedding_model_device: str = "cuda:0",
):
    """
    Load the AI News Bot.


    Args:
        env_file_path (str): Path to the environment file.
        logging_config_path (str): Path to the logging configuration file.
        model_cache_dir (str, optional): Path to the directory where the model cache is stored.
        embedding_model_device (str): Device to use for the embedding model.

    Returns:
        AINewsBot: An instance of the AINewsBot class.
    """

    from ai_news_bot import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from ai_news_bot import utils
    from ai_news_bot.langchain_bot import AINewsBot

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    bot = AINewsBot(
        model_cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        embedding_model_device=embedding_model_device,
    )

    return bot


# === Bot Runners ===


# *** Beam Endpoint ***
@endpoint(
    gpu=["T4", "A10G"],
    cpu=4,
    memory="32Gi",
    image=Image(python_version="python3.10", python_packages="requirements.txt"),
    volumes=[
        Volume(mount_path="./model_cache", name="model_cache"),
    ],
)
def run_beam(**inputs):
    """
    Run the bot under the Beam RESTful API endpoint.

    Args:
        question (str): A string containing the user's question.
        history (List[Tuple[str, str]], optional): A list of tuples containing the user's previous questions
            and the bot's responses. Defaults to None.

    Returns:
        str: The bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir="./model_cache")

    inputs = {
        "question": inputs["question"],
        "history": inputs.get("history", None),
        "context": bot,
    }

    response = _run(**inputs)

    return response


def run_local(
    question: str,
    history: Optional[list[tuple[str, str]]],
):
    """
    Run the bot locally.

    Args:
        question (str): A string containing the user's question.
        history (List[Tuple[str, str]], optional): A list of tuples containing the user's previous questions
            and the bot's responses. Defaults to None.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)

    inputs = {
        "question": question,
        "history": history,
        "context": bot,
    }

    response = _run(**inputs)

    return response


def _run(**inputs):
    """
    Central function that calls the bot and returns the response.

    Args:
        inputs (dict): A dictionary containing the following keys:
            - context: The bot instance.
            - question (str): The user's question.
            - history (list): A list of previous conversations (optional).

    Returns:
        str: The bot's response to the user's question.
    """

    bot = inputs["context"]
    input_payload = {
        "question": inputs["question"],
        "to_load_history": inputs["history"] if "history" in inputs else [],
    }
    response = bot.answer(**input_payload)

    return response


if __name__ == "__main__":

    fire.Fire(run_local)
