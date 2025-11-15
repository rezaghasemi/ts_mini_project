import yaml
from pathlib import Path
from get_logger import get_logger

logger = get_logger(__name__)


def get_config(path: str):
    CONFIG_PATH = Path(path)
    if not CONFIG_PATH.exists():
        logger.error(f"Config file {CONFIG_PATH} not found")
        raise FileNotFoundError(f"Config file {CONFIG_PATH} not found")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config file {CONFIG_PATH} loaded")
    return config
