import logging
import sys
from logging.handlers import RotatingFileHandler

# formatter
log_format = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)d - %(funcName)s - %(message)s"
formatter = logging.Formatter(fmt=log_format)

# stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)


def get_logger(name, log_path="main.log", console=True):
    """
    Simple logging wrapper that returns logger
    configured to log into file and console.

    Args:
        name (str): name of logger
        log_path (str): path of log file
        console (bool): whether to log on console

    Returns:
        logger (logging.Logger): configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ensure that logging handlers are not duplicated
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # rotating file handler
    if log_path:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 2 ** 20,  # 10 MB
            backupCount=1,  # 1 backup
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # console handler
    if console:
        logger.addHandler(stdout_handler)

    if not logger.hasHandlers():
        logger.addHandler(logging.NullHandler())

    return logger
