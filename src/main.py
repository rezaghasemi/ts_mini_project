from data_ingestion import DataIngestion
from data_preprocess import DataPreprocessing

if __name__ == "__main__":
    data_ingestion = DataIngestion("src/config/config.yaml")
    data_ingestion.run()
    data_preprocess = DataPreprocessing("src/config/config.yaml")
    data_preprocess.run()
