# miner/utils/logger.py

import logging
from datetime import datetime


NOW = datetime.now()
LOG = "logs/" + NOW.strftime("%Y-%m-%d") + ".log"
FORMAT = (
    "[%(asctime)s]"
    + " [%(filename)s:%(lineno)s - %(funcName)s]"
    + "\t%(levelname)s: %(message)s"
)


def logging_config():
    """Configures the `logging` module: creates a `log` file with the current
    date. If the file already exists, the logs are added afterwards. Only
    warning and error messages are displayed in the console.
    """
    logging.basicConfig(
        filename=LOG,
        filemode="a",
        level=logging.INFO,
        format=FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.getLogger("").addHandler(console)

