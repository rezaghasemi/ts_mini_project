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


class ARIMA(forcastingBaseModel):
    def __init__(self, config: str):
        super().__init__()
        self.config = get_config(config)

    def train(self):
        train_data, val_data = self.load_data()
        mlflow.set_experiment("ARIMA_experiment")
        with mlflow.start_run():
            p, d, q = (
                self.config["ARIMA_model"]["p"],
                self.config["ARIMA_model"]["d"],
                self.config["ARIMA_model"]["q"],
            )
            model = ARIMA(train_data, order=(p, d, q))
            model_fit = model.fit()
            evaluation = self.evaluate(model_fit, val_data)
            mlflow.log_params({"p": p, "d": d, "q": q})
            mlflow.log_metrics(evaluation)
            mlflow.log_artifact(
                Path(self.config["model_training"]["model_save_path"])
                / "arima_model.pkl"
            )

            self.save_model(model_fit)

    def predict(self):
        pass

    def evaluate(self, model_fit, val_data):
        predictions = model_fit.forecast(steps=len(val_data))
        mse = np.mean((predictions - val_data.values.flatten()) ** 2)
        return {"MSE": mse}

    def save_model(self):
        save_path = Path(self.config["model_training"]["model_save_path"])
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "arima_model.pkl"
        joblib.dump(self.model, model_file)

    def load_model(self):
        pass

    def load_data(self):
        PROC_DATA_PATH = Path(self.config["data_preprocessing"]["store_path"])
        train_data_path = PROC_DATA_PATH / "train_data.csv"
        val_data_path = PROC_DATA_PATH / "val_data.csv"
        if not train_data_path.exists() or not val_data_path.exists():
            raise FileNotFoundError("Preprocessed data files not found.")
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)

        return train_data, val_data
