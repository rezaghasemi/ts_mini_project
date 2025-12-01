import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMA

from statsmodels.tsa.stattools import adfuller
from utils.config_reader import get_config
import warnings
from interfaces.base_models import forcastingBaseModel
from pathlib import Path
import mlflow
import joblib

warnings.filterwarnings("ignore")


class SARIMA_model(forcastingBaseModel):
    def __init__(self, config: str):
        super().__init__()
        self.config = get_config(config)

    def run(self):
        self.train_data, self.val_data = self.load_data()
        mlflow.set_experiment("SARIMA_experiment")
        with mlflow.start_run():
            p, d, q, P, D, Q, seasonal_period = (
                self.config["SARIMA_model"]["p"],
                self.config["SARIMA_model"]["d"],
                self.config["SARIMA_model"]["q"],
                self.config["SARIMA_model"]["seasonal_P"],
                self.config["SARIMA_model"]["seasonal_D"],
                self.config["SARIMA_model"]["seasonal_Q"],
                self.config["SARIMA_model"]["seasonal_period"],
            )
            model = SARIMA(
                self.train_data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
            )
            self.model_fit = model.fit()

            mlflow.log_params(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "seasonal_P": P,
                    "seasonal_D": D,
                    "seasonal_Q": Q,
                    "seasonal_period": seasonal_period,
                }
            )

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
        plt.title("SARIMA Model - Actual vs Forecasted")
        plt.legend()
        path = (
            Path(self.config["model_training"]["figures_save_path"])
            / f"SARIMA_{self.config['data_ingestion']['data_set_name']}.png"
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
        max_prediction_length = self.config["SARIMA_model"]["max_prediction_length"]
        all_data = all_data["Passengers"].to_numpy()
        train_data = all_data[:-max_prediction_length]
        val_data = all_data[-max_prediction_length:]
        return train_data, val_data


if __name__ == "__main__":
    SARIMA_model = SARIMA_model(config="src/config/config.yaml")
    SARIMA_model.run()
