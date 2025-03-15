from dagster import (
    AssetSelection,
    DailyPartitionsDefinition,
    DefaultScheduleStatus,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
    load_assets_from_modules,
)

from . import assets
from .constants import START_DATE

all_assets = load_assets_from_modules([assets])

# Devide the workflows by day
daily_partitions_def = DailyPartitionsDefinition(start_date=START_DATE)

tech_news_job = define_asset_job(
    name="tech_news_extraction",
    selection=AssetSelection.all(),
    partitions_def=daily_partitions_def,
)

# Set the schedule to run every day at midnight UTC, running by default
tech_news_schedule = ScheduleDefinition(
    job=tech_news_job,
    cron_schedule="0 0 * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)

defs = Definitions(
    assets=all_assets,
    schedules=[tech_news_schedule],
)
