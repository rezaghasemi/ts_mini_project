import logging
import datetime
from pathlib import Path


LOG_DIR = "logs"
LOG_FILE_PATH = (
    Path(LOG_DIR) / f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
)

LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

basicConfig = logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    filemode="w",
)


def get_logger(name):
    return logging.getLogger(name)
