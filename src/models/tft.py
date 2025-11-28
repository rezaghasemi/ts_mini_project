from interfaces.base_models import forcastingBaseModel
from pytorch_forecasting import GroupNormalizer
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
import torch

logger = get_logger(__name__)


class TFTModel(forcastingBaseModel):
    def __init__(self, config: str):
        super().__init__()
        self.config = get_config(config)
        self.device = self.config["TFT_model"]["device"]
        self.batch_size = self.config["TFT_model"]["batch_size"]
        self.train_data_loader, self.val_data_loader = self.load_data()

        self.model = TemporalFusionTransformer.from_dataset(
            self.train_data_loader.dataset,
            learning_rate=self.config["TFT_model"]["learning_rate"],
            hidden_size=32,
            dropout=0.1,
            attention_head_size=4,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4,
        )

    def load_data(self):
        path_to_data = (
            Path(self.config["data_ingestion"]["data_set_store_path"])
            / f"{self.config['data_ingestion']['data_set_name']}.csv"
        )
        if not path_to_data.exists():
            logger.error(f"Data file not found at {path_to_data}")
            raise FileNotFoundError(f"Data file not found at {path_to_data}")
        data = pd.read_csv(path_to_data)
        data = data.assign(
            dt_month=pd.to_datetime(data["Month"], format="%Y-%m").dt.month,
            dt_year=pd.to_datetime(data["Month"], format="%Y-%m").dt.year,
        )
        data["time_idx"] = range(len(data))
        data["group_id"] = "0"  # single time series
        max_encoder_length = self.config["TFT_model"]["max_encoder_length"]
        max_prediction_length = self.config["TFT_model"]["max_prediction_length"]
        training_cutoff = data["time_idx"].max() - max_prediction_length

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="Passengers",
            group_ids=["group_id"],  # only one time series in this dataset
            min_encoder_length=max_encoder_length
            // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            # static_categoricals=["name of static categorical variables"], we don't have any in this dataset
            # static_reals=["name of static real variables"], # we don't have any in this dataset
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx", "dt_month", "dt_year"],
            time_varying_unknown_categoricals=[],  # we don't have any in this dataset
            time_varying_unknown_reals=["Passengers"],
            target_normalizer=GroupNormalizer(
                groups=["group_id"], center=True, scale_by_group=True
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        validation = TimeSeriesDataSet.from_dataset(
            training, data, predict=True, stop_randomization=True
        )

        # create dataloaders for model
        train_dataloader = training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )
        return train_dataloader, val_dataloader

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
        trainer.fit(self.model, self.train_data_loader, self.val_data_loader)
        self.save_model()
        self.plot_prediction_and_forecast()

    def plot_prediction_and_forecast(self):
        raw_predictions = self.model.predict(
            self.val_data_loader,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator="cpu"),
        )

        self.model.plot_prediction(
            raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True
        )

        path = (
            Path(self.config["model_training"]["figures_save_path"])
            / f"{self.config['model_training']['model_name']}_{self.config['data_ingestion']['data_set_name']}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()

    def save_model(self):
        pass


if __name__ == "__main__":
    model = TFTModel("src/config/config.yaml")
    model.run()
