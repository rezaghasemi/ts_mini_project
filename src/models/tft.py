from interfaces.base_models import forcastingBaseModel
from utils.config_reader import get_config
from pathlib import Path
from utils.get_logger import get_logger
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

logger = get_logger(__name__)


class TFTModel(forcastingBaseModel):
    def __init__(self, config: str):
        super().__init__()
        self.config = get_config(config)
        self.device = self.config["TFT_model"]["device"]

        train_data, test_data = self.load_data()

        self.model = TemporalFusionTransformer.from_dataset(
            train_data,
            learning_rate=self.config["TFT_model"]["learning_rate"],
            hidden_size=32,
            dropout=0.1,
            attention_head_size=4,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4,
        )
        self.train_data_dataloader = train_data.to_dataloader(
            train=True,
            batch_size=self.config["TFT_model"]["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        self.test_data_dataloader = test_data.to_dataloader(
            train=False,
            batch_size=self.config["TFT_model"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )

    def load_data(self):
        path_to_data = (
            Path(self.config["data_preprocessing"]["store_path"])
            / self.config["data_ingestion"]["data_set_name"]
        )

        self.train_raw_data = pd.read_csv(path_to_data / "train_data.csv")
        self.test_raw_data = pd.read_csv(path_to_data / "test_data.csv")

        history_length = self.config["TFT_model"]["history_length"]
        prediction_length = self.config["TFT_model"]["prediction_length"]

        self.train_raw_data["group_id"] = 0
        self.train_raw_data["time_idx"] = range(len(self.train_data))
        self.test_raw_data["group_id"] = 0
        self.test_raw_data["time_idx"] = range(len(self.test_data))

        self.all_data = pd.concat([self.train_raw_data, self.test_raw_data])[
            ["Date", "Year", "Month", "Passengers"]
        ]

        train_DataSet = TimeSeriesDataSet(
            self.train_raw_data,
            group_ids=["group_id"],
            target="Passengers",
            time_idx="time_idx",
            min_encoder_length=history_length,
            max_encoder_length=history_length,
            max_prediction_length=prediction_length,
            time_varying_known_reals=["Month", "Year"],
            time_varying_unknown_reals=["Passengers"],
        )

        test_DataSet = TimeSeriesDataSet(
            self.test_raw_data,
            group_ids=["group_id"],
            target="Passengers",
            time_idx="time_idx",
            min_encoder_length=history_length,
            max_encoder_length=history_length,
            max_prediction_length=prediction_length,
            time_varying_known_reals=["Month", "Year"],
            time_varying_unknown_reals=["Passengers"],
        )

        return train_DataSet, test_DataSet

    def run(self):
        epochs = self.config["TFT_model"]["epochs"]
        save_path = Path(self.config["model_training"]["model_save_path"]) / "tft"
        mlflow_logger = MLFlowLogger(
            experiment_name="my_experiment",
            tracking_uri="file:./mlruns",  # local folder; can also be a remote server
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            filename="best-tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,  # number of passes through the data
            gradient_clip_val=0.1,  # stabilizes training
            callbacks=[checkpoint_callback],
            accelerator=self.device,
            devices=1,
            logger=mlflow_logger,
        )
        trainer.fit(
            self.model,
            self.train_data_dataloader,
            self.test_data_dataloader,
        )
        self.save_model()
        self.plot_prediction_and_forecast()

    def plot_prediction_and_forecast(self):
        pass

    def ar_forecast(sel, input_data: pd.DataFrame, steps: int) -> pd.DataFrame:
        pass

    def save_model(self):
        pass


if __name__ == "__main__":
    model = TFTModel("src/config/config.yaml")
    model.run()
