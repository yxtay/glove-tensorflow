import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

# init root logger with null handler
logging.basicConfig(handlers=[logging.NullHandler()])

# init log queue for handler and listener
log_queue = queue.Queue()


def __get_log_formatter() -> logging.Formatter:
    # formatter
    log_format = " - ".join([
        "%(asctime)s",
        "%(levelname)s",
        "%(name)s",
        "%(processName)s",
        "%(threadName)s",
        "%(filename)s",
        "%(module)s",
        "%(lineno)d",
        "%(funcName)s",
        "%(message)s",
    ])
    date_format = "%Y-%m-%dT%H:%M:%S"
    log_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    return log_formatter


def __get_file_handler(log_path: str = "main.log") -> RotatingFileHandler:
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 2 ** 20, backupCount=1  # 10 MB  # 1 backup
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(__get_log_formatter())
    return file_handler


def __get_stdout_handler() -> logging.StreamHandler:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(__get_log_formatter())
    return stdout_handler


def configure_log_handlers(console: bool = True, log_path: str = "main.log") -> QueueListener:
    """
    Configure log queue listener to log into file and console.

    Args:
        console (bool): whether to log on console
        log_path (str): path of log file

    Returns:
        log_qlistener (logging.handlers.QueueListener): configured log queue listener
    """
    global log_qlistener
    try:
        log_qlistener.stop()
    except (AttributeError, NameError):
        pass

    handlers = []

    # rotating file handler
    if log_path:
        file_handler = __get_file_handler(log_path)
        handlers.append(file_handler)

    # console handler
    if console:
        stdout_handler = __get_stdout_handler()
        handlers.append(stdout_handler)

    log_qlistener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    log_qlistener.start()
    return log_qlistener


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Simple logging wrapper that returns logger
    configured to log into file and console.

    Args:
        name (str): name of logger

    Returns:
        logger (logging.Logger): configured logger
    """
    logger = logging.getLogger(name)
    for log_handler in logger.handlers[:]:
        logger.removeHandler(log_handler)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(QueueHandler(log_queue))

    return logger


configure_log_handlers()
logger = get_logger()
