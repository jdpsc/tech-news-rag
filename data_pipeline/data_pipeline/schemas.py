import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.staging.huggingface import chunk_by_attention_window

from data_pipeline.embeddings import EmbeddingModel


class Document(BaseModel):
    """
    A Pydantic model representing a document.

    Attributes:
        id (str): The ID of the document.
        group_key (Optional[str]): The group key of the document.
        metadata (dict): The metadata of the document.
        text (list): The text of the document.
        chunks (list): The chunks of the document.
        embeddings (list): The embeddings of the document.

    Methods:
        to_payloads: Returns the payloads of the document.
        compute_chunks: Computes the chunks of the document.
        compute_embeddings: Computes the embeddings of the document.
    """

    id: str
    group_key: Optional[str] = None
    metadata: dict = {}
    text: list = []
    chunks: list = []
    embeddings: list = []

    def to_payloads(self) -> tuple[list[str], list[dict]]:
        """
        Returns the payloads of the document.

        Returns:
            tuple[list[str], list[dict]]: A tuple containing the IDs and payloads of the document.
        """

        payloads = []
        ids = []
        for chunk in self.chunks:
            payload = self.metadata
            payload.update({"text": chunk})
            # Create the chunk ID using the hash of the chunk to avoid storing duplicates.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return ids, payloads

    def compute_chunks(self, model: EmbeddingModel) -> "Document":
        """
        Computes the chunks of the document.

        Args:
            model (EmbeddingModel): The embedding model to use for computing the chunks.

        Returns:
            Document: The document object with the computed chunks.
        """

        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)

        return self

    def compute_embeddings(self, model: EmbeddingModel) -> "Document":
        """
        Computes the embeddings for each chunk in the document using the specified embedding model.

        Args:
            model (EmbeddingModel): The embedding model to use for computing the embeddings.

        Returns:
            Document: The document object with the computed embeddings.
        """

        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self


class ArxivArticle(Document):
    """
    Represents an Arxiv article. Docs: "https://info.arxiv.org/help/api/index.html"

    Attributes:
        id (int): News article ID
        title (str): itle of the article
        summary (str): Summary text for the article
        authors (List[str]): Original authors of the article
        published_at (datetime): Date article was published
        updated_at (datetime): Date article was updated
        url (str): URL of article
        tags (List[str]): List of tags
    """

    @staticmethod
    def to_document(raw_payload: dict) -> Document:
        """
        Converts the Arxiv article to a Document object.

        Returns:
            Document: A Document object representing the Arxiv article.
        """

        document_id = hashlib.md5(raw_payload["title"].encode()).hexdigest()
        document = Document(id=document_id)

        title = clean_non_ascii_chars(
            replace_unicode_quotes(clean(raw_payload["title"]))
        )
        summary = clean_non_ascii_chars(
            replace_unicode_quotes(clean(raw_payload["summary"]))
        )
        url = raw_payload["link"]
        published_at = datetime.strptime(raw_payload["published"], "%Y-%m-%dT%H:%M:%SZ")
        updated_at = datetime.strptime(raw_payload["updated"], "%Y-%m-%dT%H:%M:%SZ")

        authors = []
        for author in raw_payload["authors"]:
            author = clean_non_ascii_chars(
                replace_unicode_quotes(clean(author["name"]))
            )
            authors.append(author)

        tags = []
        for tag in raw_payload["tags"]:
            tag = clean_non_ascii_chars(replace_unicode_quotes(clean(tag["term"])))
            tags.append(tag)

        # The information that will be embedded
        document.text = [title, summary]

        document.metadata["title"] = title
        document.metadata["summary"] = summary
        document.metadata["authors"] = authors
        document.metadata["published_at"] = published_at
        document.metadata["updated_at"] = updated_at
        document.metadata["url"] = url
        document.metadata["tags"] = tags

        return document
