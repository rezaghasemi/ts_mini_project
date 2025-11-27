from data_ingestion import DataIngestion
from data_preprocess import DataPreprocessing
from utils.get_model import get_model

if __name__ == "__main__":
    data_ingestion = DataIngestion("src/config/config.yaml")
    data_ingestion.run()

    # model = get_model("src/config/config.yaml")
    # model.train()
