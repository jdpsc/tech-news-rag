from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json
from training_pipeline.prompt_templates.prompter import get_llm_template


@dataclass(frozen=True)
class DataSample:
    """
    A data sample for a question answering model.

    Attributes:
        context (str): The context for the question (from the DB).
        chat_history (str): The chat history for the question.
        question (str): The question to be answered.
        answer (str): The answer to the question.
    """

    context: str = ""
    chat_history: str = ""
    question: str = ""
    answer: str = ""


class InstructionDataset:
    def __init__(
        self,
        data_path: Path,
        scope: Scope = Scope.TRAINING,
        template: str = "falcon",
        max_samples: Optional[int] = None,
    ):
        """
        A class representing a Q&A dataset.

        Args:
            data_path (Path): The path to the data file.
            scope (Scope, optional): The scope of the dataset. Defaults to Scope.TRAINING.
            template (str, optional): The template to use for the dataset. Defaults to "falcon".
            max_samples (Optional[int], optional): The maximum number of samples to use. Defaults to None.
        """

        self._data_path = data_path
        self._scope = scope
        self._max_samples = max_samples
        self._template = get_llm_template(template)
        self._raw_data = self.load(data_path)

    def load(self, data_path: Path) -> list[DataSample]:
        """
        Loads the data from the specified path.

        Args:
            data_path (Path): The path to the data file.

        Returns:
            List[DataSample]: The loaded data.
        """

        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[: self._max_samples]

        return self.deserialize(data)

    def deserialize(self, data: list[dict]) -> list[DataSample]:
        """
        Deserializes the data.

        Args:
            data (List[dict]): The data to deserialize.

        Returns:
            List[DataSample]: The deserialized data.
        """

        if self._scope == Scope.TRAINING:
            return [
                DataSample(
                    context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                    answer=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                )
                for sample in data
            ]

    def to_huggingface(self) -> Dataset:
        """
        Preprocesses the data & returns a HuggingFace dataset.

        Returns:
            Dataset: The HuggingFace dataset.
        """

        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            template_mapping_func = self._template.format_train
        else:
            template_mapping_func = self._template.format_infer

        dataset = dataset.map(self.clean)
        dataset = dataset.map(
            template_mapping_func, remove_columns=dataset.column_names
        )

        return dataset

    def clean(self, samples: dict[str, str]) -> dict[str, str]:
        """
        Cleans the samples.

        Args:
            samples (dict[str, str]): The samples to clean.

        Returns:
            dict[str, str]: The cleaned samples.
        """

        for key, sample in samples.items():
            cleaned_sample = clean_extra_whitespace(sample)
            cleaned_sample = group_broken_paragraphs(cleaned_sample)

            samples[key] = cleaned_sample

        return samples
