import pandas as pd
from dagster import AssetExecutionContext, Config, DailyPartitionsDefinition, asset

from data_pipeline.database import QdrantWriter
from data_pipeline.embeddings import EmbeddingModel
from data_pipeline.sources import ArvixSource

from .constants import NEWS_QUERY, START_DATE, TOP_NEWS_LIMIT

daily_partitions_def = DailyPartitionsDefinition(start_date=START_DATE)


class TechNewsConfig(Config):
    top_news_limit: int = TOP_NEWS_LIMIT  # The number of top news to get per day
    news_query: str = NEWS_QUERY  # The query to search for


# Define the asset, partitioned by day
@asset(partitions_def=daily_partitions_def)
def tech_news_embedded_and_write(
    context: AssetExecutionContext, config: TechNewsConfig
):

    partition_date_str = context.partition_key

    start_datetime = pd.to_datetime(partition_date_str)
    end_datetime = start_datetime + pd.Timedelta(days=1)

    # Get the data
    sources = [ArvixSource(max_results=config.top_news_limit, query=config.news_query)]
    documents = []
    for source in sources:
        document = source.get_batch_data(
            from_datetime=start_datetime, to_datetime=end_datetime
        )
        documents.extend(document)

    # Get the model
    model = EmbeddingModel()

    # Process the data - chunking and enmbeddings
    for document in documents:
        document.compute_chunks(model)
        document.compute_embeddings(model)

    # Write to Qdrant
    writer = QdrantWriter()
    writer.write_batch(documents)
