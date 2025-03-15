import argparse
import logging
from threading import Thread
from typing import Generator

import gradio as gr

from tools.bot import load_bot

logger = logging.getLogger(__name__)


def parseargs() -> argparse.Namespace:
    """
    Parses command line arguments for the AI News Bot.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(description="AI News Bot")

    parser.add_argument(
        "--env-file-path",
        type=str,
        default=".env",
        help="Path to the environment file",
    )

    parser.add_argument(
        "--logging-config-path",
        type=str,
        default="logging.yaml",
        help="Path to the logging configuration file",
    )

    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="./model_cache",
        help="Path to the directory where the model cache will be stored",
    )

    parser.add_argument(
        "--embedding-model-device",
        type=str,
        default="cuda:0",
        help="Device to use for the embedding model (e.g. 'cpu', 'cuda:0', etc.)",
    )

    return parser.parse_args()


args = parseargs()


bot = load_bot(
    env_file_path=args.env_file_path,
    logging_config_path=args.logging_config_path,
    model_cache_dir=args.model_cache_dir,
    embedding_model_device=args.embedding_model_device,
)


# === Gradio Interface ===


def predict(message: str, history: list[list[str]]) -> Generator[str, None, None]:
    """
    Predicts a response to a given message using the ai_news_bot Gradio UI.

    Args:
        message (str): The message to generate a response for.
        history (list[list[str]]): A list of previous conversations.

    Returns:
        str: The generated response.
    """

    generate_kwargs = {
        "question": message,
        "to_load_history": history,
    }

    if bot.is_streaming:
        t = Thread(target=bot.answer, kwargs=generate_kwargs)
        t.start()

        for partial_answer in bot.stream_answer():
            yield partial_answer
    else:
        yield bot.answer(**generate_kwargs)


demo = gr.ChatInterface(
    predict,
    textbox=gr.Textbox(
        placeholder="Ask me AI question",
        label="AI question",
        container=False,
        scale=7,
    ),
    title="Your Personal AI Assistant",
    description="Ask me any question about AI, and I will do my best to answer them.",
    theme="soft",
    examples=[
        [
            "Can you give me examples of RAG retrieval and post-retrieval optimization techniques?",
        ],
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
