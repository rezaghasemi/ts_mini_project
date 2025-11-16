from utils.config_reader import get_config
from utils.get_logger import get_logger
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = get_logger(__name__)


class DataPreprocessing:
    def __init__(self, config: str):
        self.config = get_config(config)

    def run(self):
        logger.info("Data Preprocessing")
        raw_data_path = Path(self.config["data_ingestion"]["data_set_store_path"])
        raw_data = self.get_raw_data(raw_data_path)
        proccesssed_raw_data = self.process_data(raw_data)
        train_data, test_data, val_data = self.split_data(proccesssed_raw_data)
        self.store_data(train_data, test_data, val_data)
        logger.info("Data Preprocessing Completed")

    def process_data(self, raw_data: pd.DataFrame) -> list[pd.DataFrame]:
        if self.config["data_ingestion"]["data_set_name"] == "airline_passengers":
            if self.config["data_preprocessing"]["min_max_scale"]:
                self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                raw_data["SUNACTIVITY"] = self.min_max_scaler.fit_transform(
                    raw_data[["SUNACTIVITY"]]
                )
                logger.info("Min Max Scaling Applied on Data into range 0 to 1")
            if self.config["data_preprocessing"]["standard_scale"]:
                self.standard_scaler = StandardScaler()
                raw_data["SUNACTIVITY"] = self.standard_scaler.fit_transform(
                    raw_data[["SUNACTIVITY"]]
                )
                logger.info("Standard Scaling Applied on Data")
            return raw_data
        else:
            logger.error("Data set not supported")
            raise Exception("Data set not supported")

    def split_data(self, proccesssed_raw_data: pd.DataFrame):
        if self.config["data_ingestion"]["data_set_name"] == "airline_passengers":
            train_test_split_ratio = self.config["data_preprocessing"][
                "train_test_split_ratio"
            ]
            train_data = proccesssed_raw_data[
                : int(len(proccesssed_raw_data) * train_test_split_ratio)
            ]
            test_data = proccesssed_raw_data[
                int(len(proccesssed_raw_data) * train_test_split_ratio) :
            ]
            return train_data, test_data, None
        else:
            logger.error("Data set not supported")
            raise Exception("Data set not supported")

    def store_data(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, val_data: pd.DataFrame
    ):
        store_path = Path(self.config["data_preprocessing"]["store_path"])
        store_path.mkdir(parents=True, exist_ok=True)
        if train_data is not None:
            train_data.to_csv(store_path / "train_data.csv", index=False)
        if test_data is not None:
            test_data.to_csv(store_path / "test_data.csv", index=False)
        if val_data is not None:
            val_data.to_csv(store_path / "val_data.csv", index=False)

    def get_raw_data(self, raw_data_path: Path):
        raw_data = pd.read_csv(raw_data_path / "raw_data.csv")
        return raw_data


if __name__ == "__main__":
    data_preprocess = DataPreprocessing("src/config/config.yaml")
    data_preprocess.run()
