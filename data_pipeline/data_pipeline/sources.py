import datetime
from abc import ABC
from typing import Optional

import feedparser as fp
import requests

from data_pipeline.schemas import ArxivArticle, Document


class DataSource(ABC):
    """
    Abstract class for data sources, should be implemented by concrete data sources.
    """

    def get_batch_data(
        self,
        from_datetime: Optional[datetime.datetime] = None,
        to_datetime: Optional[datetime.datetime] = None,
    ) -> list[Document]:
        raise NotImplementedError


class ArvixSource(DataSource):
    """
    Data source for Arvix articles. Docs: "https://info.arxiv.org/help/api/index.html"

    Args:
        max_results (int): Maximum number of results to return
        query (str): Query string to search for
    """

    base_url = "http://export.arxiv.org/api/query"
    sortBy = "submittedDate"
    sortOrder = "descending"

    def __init__(self, max_results: int, query: str) -> None:
        super().__init__()

        self.max_results = max_results
        self.query = query

    def get_batch_data(
        self,
        from_datetime: Optional[datetime.datetime] = None,
        to_datetime: Optional[datetime.datetime] = None,
    ) -> list[Document]:
        """
        Get a batch of Arvix articles from the API from from_datetime to to_datetime.
        """

        params: dict[str, str | int] = {
            "search_query": f"all:{self.query}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": self.sortBy,
            "sortOrder": self.sortOrder,
        }

        if from_datetime:
            params["start_date"] = int(from_datetime.timestamp())
        if to_datetime:
            params["end_date"] = int(to_datetime.timestamp())

        # TODO: Add error handling for requests
        response = requests.get(self.base_url, params=params)
        payload = response.text

        # Convert XML to JSON
        parsed = fp.parse(payload)
        entries = parsed["entries"]

        # Parse entries into documents using the ArxivArticle schema
        documents = [ArxivArticle.to_document(entry) for entry in entries]

        return documents
