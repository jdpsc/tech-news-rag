import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ai_news_bot.embeddings import EmbeddingModelSingleton
import torch
from transformers import BatchEncoding


@pytest.fixture
def mock_embedding_model():
    with patch("ai_news_bot.embeddings.AutoModel") as MockModel:
        mock_model = MockModel.from_pretrained.return_value
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        yield EmbeddingModelSingleton()


def test_embedding_model_initialization(mock_embedding_model):
    assert mock_embedding_model.max_input_length > 0
    assert mock_embedding_model.tokenizer is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_embedding_model_call(mock_embedding_model):
    mock_embedding_model._tokenizer = MagicMock()
    mock_embedding_model._tokenizer.return_value = BatchEncoding(
        {
            "input_ids": torch.tensor([[1] * mock_embedding_model.max_input_length]),
            "attention_mask": torch.tensor(
                [[1] * mock_embedding_model.max_input_length]
            ),
        }
    )
    mock_embedding_model._model = MagicMock()
    mock_embedding_model._model.return_value.last_hidden_state = np.random.rand(
        1, 10, 768
    )

    input_text = "Test input"
    embeddings = mock_embedding_model(input_text)

    assert isinstance(embeddings, list)
    assert len(embeddings) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_embedding_model_truncation(mock_embedding_model):
    long_text = "word " * (mock_embedding_model.max_input_length + 10)
    mock_embedding_model._tokenizer = MagicMock()
    mock_embedding_model._tokenizer.return_value = BatchEncoding(
        {
            "input_ids": torch.tensor([[1] * mock_embedding_model.max_input_length]),
            "attention_mask": torch.tensor(
                [[1] * mock_embedding_model.max_input_length]
            ),
        }
    )
    mock_embedding_model._model = MagicMock()
    mock_embedding_model._model.return_value.last_hidden_state = np.random.rand(
        1, 10, 768
    )

    embeddings = mock_embedding_model(long_text)

    assert isinstance(embeddings, list)
    assert len(embeddings) > 0


def test_embedding_model_device(mock_embedding_model):
    assert mock_embedding_model._device in ["cpu"] + [f"cuda:{i}" for i in range(8)]
