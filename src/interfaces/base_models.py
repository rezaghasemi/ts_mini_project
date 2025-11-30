from abc import ABC, abstractmethod


class forcastingBaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot_prediction_and_forecast(self):
        pass

    @abstractmethod
    def load_data(self):
        pass
