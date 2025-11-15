from interfaces.base_models import forcastingBaseModel


class TFTModel(forcastingBaseModel):
    def __init__(self, config: str):
        super().__init__()

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def load_data(self):
        pass


if __name__ == "__main__":
    model = TFTModel("config/config.yaml")
