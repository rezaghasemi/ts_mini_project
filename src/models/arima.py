import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from utils.config_reader import get_config
import warnings
from interfaces.base_models import forcastingBaseModel
from pathlib import Path
import mlflow
import joblib

warnings.filterwarnings("ignore")


class ARIMA_model(forcastingBaseModel):
    def __init__(self, config: str):
        super().__init__()
        self.config = get_config(config)

    def run(self):
        self.train_data, self.val_data = self.load_data()
        mlflow.set_experiment("ARIMA_experiment")
        with mlflow.start_run():
            p, d, q = (
                self.config["ARIMA_model"]["p"],
                self.config["ARIMA_model"]["d"],
                self.config["ARIMA_model"]["q"],
            )
            model = ARIMA(self.train_data, order=(p, d, q))
            self.model_fit = model.fit()

            mlflow.log_params({"p": p, "d": d, "q": q})

            self.plot_prediction_and_forecast()

    def plot_prediction_and_forecast(self):
        result = self.model_fit.forecast(steps=len(self.val_data))
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(self.val_data) + 1),
            self.val_data,
            color="blue",
            label="Actual",
        )
        plt.plot(
            range(1, len(self.val_data) + 1),
            result,
            color="red",
            label="Forecasted",
        )
        plt.xlabel("Month")
        plt.ylabel("Number of Passengers")
        plt.title("ARIMA Model - Actual vs Forecasted")
        plt.legend()
        path = (
            Path(self.config["model_training"]["figures_save_path"])
            / f"ARIMA_{self.config['data_ingestion']['data_set_name']}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()

    def load_model(self):
        pass

    def load_data(self):
        RAW_DATA_PATH = Path(self.config["data_ingestion"]["data_set_store_path"])
        data_set_name = self.config["data_ingestion"]["data_set_name"]
        all_data = pd.read_csv(RAW_DATA_PATH / f"{data_set_name}.csv")
        max_prediction_length = self.config["ARIMA_model"]["max_prediction_length"]
        all_data = all_data["Passengers"].to_numpy()
        train_data = all_data[:-max_prediction_length]
        val_data = all_data[-max_prediction_length:]
        return train_data, val_data


if __name__ == "__main__":
    arima_model = ARIMA_model(config="src/config/config.yaml")
    arima_model.run()
