import os
import pytest
from unittest.mock import patch, MagicMock
from ai_news_bot.qdrant import build_qdrant_client


def test_build_qdrant_client_with_arguments():
    url = "http://test-qdrant-url.com"
    api_key = "test-api-key"

    with patch("ai_news_bot.qdrant.qdrant_client.QdrantClient") as MockQdrantClient:
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        client = build_qdrant_client(url=url, api_key=api_key)

        MockQdrantClient.assert_called_once_with(url, api_key=api_key)
        assert client == mock_client


def test_build_qdrant_client_with_env_variables():
    url = "http://env-qdrant-url.com"
    api_key = "env-api-key"

    with patch.dict(os.environ, {"QDRANT_URL": url, "QDRANT_API_KEY": api_key}), patch(
        "ai_news_bot.qdrant.qdrant_client.QdrantClient"
    ) as MockQdrantClient:
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client

        client = build_qdrant_client()

        MockQdrantClient.assert_called_once_with(url, api_key=api_key)
        assert client == mock_client


def test_build_qdrant_client_missing_url():
    api_key = "test-api-key"

    with patch.dict(os.environ, {"QDRANT_API_KEY": api_key}, clear=True):
        with pytest.raises(
            KeyError,
            match="QDRANT_URL must be set as environment variable or manually passed as an argument.",
        ):
            build_qdrant_client()


def test_build_qdrant_client_missing_api_key():
    url = "http://test-qdrant-url.com"

    with patch.dict(os.environ, {"QDRANT_URL": url}, clear=True):
        with pytest.raises(
            KeyError,
            match="QDRANT_API_KEY must be set as environment variable or manually passed as an argument.",
        ):
            build_qdrant_client()
