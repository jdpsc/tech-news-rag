import logging
from pathlib import Path
from typing import Iterable

from langchain import chains
from langchain.memory import ConversationBufferWindowMemory

from ai_news_bot import constants
from ai_news_bot.chains import (
    ContextExtractorChain,
    AINewsBotQAChain,
    StatelessMemorySequentialChain,
)
from ai_news_bot.embeddings import EmbeddingModelSingleton
from ai_news_bot.models import build_huggingface_pipeline
from ai_news_bot.qdrant import build_qdrant_client
from ai_news_bot.template import get_llm_template

logger = logging.getLogger(__name__)


class AINewsBot:
    """
    A language chain bot that uses a language model to generate responses to user inputs.

    Args:
        llm_model_id (str): The ID of the Hugging Face language model to use.
        llm_qlora_model_id (str): The ID of the Hugging Face QLora model to use.
        llm_template_name (str): The name of the LLM template to use.
        llm_inference_max_new_tokens (int): The maximum number of new tokens to generate during inference.
        llm_inference_temperature (float): The temperature to use during inference.
        vector_collection_name (str): The name of the Qdrant vector collection to use.
        vector_db_search_topk (int): The number of nearest neighbors to search for in the Qdrant vector database.
        model_cache_dir (Path): The directory to use for caching the language model and embedding model.
        streaming (bool): Whether to use the Hugging Face streaming API for inference.
        embedding_model_device (str): The device to use for the embedding model.

    Attributes:
        aibot_chain (Chain): The language chain that generates responses to user inputs.
    """

    def __init__(
        self,
        llm_model_id: str = constants.LLM_MODEL_ID,
        llm_qlora_model_id: str = constants.LLM_QLORA_CHECKPOINT,
        llm_template_name: str = constants.TEMPLATE_NAME,
        llm_inference_max_new_tokens: int = constants.LLM_INFERNECE_MAX_NEW_TOKENS,
        llm_inference_temperature: float = constants.LLM_INFERENCE_TEMPERATURE,
        vector_collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        vector_db_search_topk: int = constants.VECTOR_DB_SEARCH_TOPK,
        model_cache_dir: Path | None = constants.CACHE_DIR,
        streaming: bool = False,
        embedding_model_device: str = "cuda:0",
    ):
        self._llm_model_id = llm_model_id
        self._llm_qlora_model_id = llm_qlora_model_id
        self._llm_template_name = llm_template_name
        self._llm_template = get_llm_template(name=self._llm_template_name)
        self._llm_inference_max_new_tokens = llm_inference_max_new_tokens
        self._llm_inference_temperature = llm_inference_temperature
        self._vector_collection_name = vector_collection_name
        self._vector_db_search_topk = vector_db_search_topk

        self._qdrant_client = build_qdrant_client()

        self._embd_model = EmbeddingModelSingleton(
            cache_dir=str(model_cache_dir) if model_cache_dir else None,
            device=embedding_model_device,
        )
        self._llm_agent, self._streamer = build_huggingface_pipeline(
            llm_model_id=llm_model_id,
            llm_lora_model_id=llm_qlora_model_id,
            max_new_tokens=llm_inference_max_new_tokens,
            temperature=llm_inference_temperature,
            use_streamer=streaming,
            cache_dir=model_cache_dir,
        )
        self.aibot_chain = self.build_chain()

    def build_chain(self) -> chains.SequentialChain:
        """
        Constructs and returns a AI news bot chain.
        This chain is designed to take as input the a 'question', and it will
        connect to the VectorDB, searches the AI news based on the user's question and injects them into the
        payload that is further passed as a prompt to a AI news fine-tuned LLM that will provide answers.

        The chain consists of two primary stages:
        1. Context Extractor: This stage is responsible for embedding the user's question,
        which means converting the textual question into a numerical representation.
        This embedded question is then used to retrieve relevant context from the VectorDB.
        The output of this chain will be a dict payload.

        2. LLM Generator: Once the context is extracted,
        this stage uses it to format a full prompt for the LLM and
        then feed it to the model to get a response that is relevant to the user's question.

        Returns
        -------
        chains.SequentialChain
            The constructed AI news bot chain.

        Notes
        -----
        The actual processing flow within the chain can be visualized as:
        [question: str] > ContextChain >
        [question:str] + [context: str] > AINewsBotQAChain >
        [answer: str]
        """

        logger.info("Building 1/3 - ContextExtractorChain")
        context_retrieval_chain = ContextExtractorChain(
            embedding_model=self._embd_model,
            vector_store=self._qdrant_client,
            vector_collection=self._vector_collection_name,
            top_k=self._vector_db_search_topk,
        )

        logger.info("Building 2/3 - AINewsBotQAChain")
        llm_generator_chain = AINewsBotQAChain(
            hf_pipeline=self._llm_agent,
            template=self._llm_template,
        )

        logger.info("Building 3/3 - Connecting chains into SequentialChain")
        seq_chain = StatelessMemorySequentialChain(
            history_input_key="to_load_history",
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                k=3,
            ),
            chains=[context_retrieval_chain, llm_generator_chain],
            input_variables=["question", "to_load_history"],
            output_variables=["answer"],
            verbose=True,
        )

        logger.info("Done building SequentialChain.")
        logger.info("Workflow:")
        logger.info(
            """
            [question: str] > ContextChain >
            [question:str] + [context: str] > AINewsBotQAChain >
            [answer: str]
            """
        )

        return seq_chain

    def answer(
        self,
        question: str,
        to_load_history: list[tuple[str, str]],
    ) -> str:
        """
        Given a question make the LLM generate a response.

        Parameters
        ----------
        question : str
            User question.

        Returns
        -------
        str
            LLM generated response.
        """

        inputs = {
            "question": question,
            "to_load_history": to_load_history if to_load_history else [],
        }
        response = self.aibot_chain.run(inputs)

        return response

    def stream_answer(self) -> Iterable[str]:
        """Stream the answer from the LLM after each token is generated after calling `answer()`."""

        assert (
            self._streamer is not None
        ), "Stream answer not available. Build the bot with `use_streamer=True`."

        partial_answer = ""
        for new_token in self._streamer:
            if new_token != self._llm_template.eos:
                partial_answer += new_token

                yield partial_answer
