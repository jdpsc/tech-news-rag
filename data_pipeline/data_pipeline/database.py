import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from data_pipeline.constants import (
    EMBEDDING_MODEL_MAX_INPUT_LENGTH,
    VECTOR_DB_OUTPUT_COLLECTION_NAME,
)
from data_pipeline.schemas import Document


class QdrantWriter:
    """
    A class for writing document embeddings to a Qdrant collection.

    Args:
        url (Optional[str], optional): The URL of the Qdrant server.
        api_key (Optional[str], optional): The API key to use for authentication.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to VECTOR_DB_OUTPUT_COLLECTION_NAME.
        vector_size (int, optional): The size of the vectors to write. Defaults
            to EMBEDDING_MODEL_MAX_INPUT_LENGTH
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = VECTOR_DB_OUTPUT_COLLECTION_NAME,
        vector_size: int = EMBEDDING_MODEL_MAX_INPUT_LENGTH,
    ):
        self.client = self.build_qdrant_client(url, api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Create the collection if it does not exist
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )

    def write(self, document: Document):
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def write_batch(self, documents: list[Document]):

        for document in documents:
            self.write(document)

    def build_qdrant_client(
        self, url: Optional[str] = None, api_key: Optional[str] = None
    ):
        """
        Builds a QdrantClient object with the given URL and API key.

        Args:
            url (Optional[str]): The URL of the Qdrant server. If not provided,
                it will be read from the QDRANT_URL environment variable.
            api_key (Optional[str]): The API key to use for authentication. If not provided,
                it will be read from the QDRANT_API_KEY environment variable.

        Raises:
            KeyError: If the QDRANT_URL or QDRANT_API_KEY environment variables are not set
                and no values are provided as arguments.

        Returns:
            QdrantClient: A QdrantClient object connected to the specified Qdrant server.
        """

        if url is None:
            try:
                url = os.environ["QDRANT_URL"]
            except KeyError:
                raise KeyError(
                    "QDRANT_URL must be set as environment variable or manually passed as an argument."
                )

        if api_key is None:
            try:
                api_key = os.environ["QDRANT_API_KEY"]
            except KeyError:
                raise KeyError(
                    "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
                )

        client = QdrantClient(url, api_key=api_key)

        return client
