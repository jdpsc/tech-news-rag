from typing import Any

import qdrant_client
from langchain import chains
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from ai_news_bot.embeddings import EmbeddingModelSingleton
from ai_news_bot.template import PromptTemplate


class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: dict[str, list[list[str]]], **kwargs) -> dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: dict[str, str],
        outputs: dict[str, str],
        return_only_outputs: bool = False,
    ) -> dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of AI news.

    Attributes:
    -----------
    top_k : int
        The number of top matches to retrieve from the vector store.
    embedding_model : EmbeddingModelSingleton
        The embedding model to use for encoding the question.
    vector_store : qdrant_client.QdrantClient
        The vector store to search for matches.
    vector_collection : str
        The name of the collection to search in the vector store.
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str

    @property
    def input_keys(self) -> list[str]:
        return ["question"]

    @property
    def output_keys(self) -> list[str]:
        return ["context"]

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question_str = inputs["question"]

        cleaned_question = self.clean(question_str)
        # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
        # pass them through the model and average the embeddings.
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)

        matches = self.vector_store.search(
            query_vector=embeddings,
            limit=self.top_k,
            collection_name=self.vector_collection,
        )

        context = ""
        for match in matches:
            context += match.payload["summary"] + "\n"

        return {
            "context": context,
        }

    def clean(self, question: str) -> str:
        """
        Clean the input question by removing unwanted characters.

        Parameters:
        -----------
        question : str
            The input question to clean.

        Returns:
        --------
        str
            The cleaned question.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class AINewsBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self) -> list[str]:
        """Returns a list of input keys for the chain"""

        return ["context"]

    @property
    def output_keys(self) -> list[str]:
        """Returns a list of output keys for the chain"""

        return ["answer"]

    def _call(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        inputs = self.clean(inputs)
        prompt = self.template.format_infer(
            {
                "context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        response = self.hf_pipeline(prompt["prompt"])

        return {"answer": response}

    def clean(self, inputs: dict[str, str]) -> dict[str, str]:
        """Cleans the inputs by removing extra whitespace and grouping broken paragraphs"""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs
