import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger with terminal and optional file output."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Default Terminal Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional File Handler
        log_file = os.getenv("LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger
