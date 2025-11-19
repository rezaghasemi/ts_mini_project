from utils.get_logger import get_logger
from models.tft import TFTModel
from utils.config_reader import get_config

logger = get_logger(__name__)


def get_model(config_path: str):
    config = get_config(config_path)
    if config["model_training"]["model_name"] == "TFT":
        return TFTModel(config_path)
    else:
        logger.error("Model not supported")
        raise Exception("Model not supported")
