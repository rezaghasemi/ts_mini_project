from get_logger import get_logger
from config_reader import get_config
import pandas as pd
from pathlib import Path

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: str):
        self.config = get_config(config)

    def get_data(self):
        if self.config["data_ingestion"]["data_set_source"] == "local":
            if self.config["data_ingestion"]["data_set_name"] == "airline_passengers":
                from statsmodels.datasets import sunspots

                raw_data = sunspots.load_pandas().data
                return raw_data
        else:
            raise Exception("Data source not supported")

    def store_data(self, raw_data: pd.DataFrame):
        raw_data_path = Path(self.config["data_ingestion"]["data_set_store_path"])
        raw_data_path.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(raw_data_path / "raw_data.csv", index=False)

    def run(self):
        logger.info("Ingesting data")
        raw_data = self.get_data()
        logger.info("Data Loaded")
        self.store_data(raw_data)
        logger.info("Data Stored")


if __name__ == "__main__":
    data_ingestion = DataIngestion("src/config/config.yaml")
    data_ingestion.run()
