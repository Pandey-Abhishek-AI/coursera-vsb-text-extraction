import logging

def get_logger(name: str = "myapp") -> logging.Logger:
    """
    Create (or retrieve) a logger configured to log DEBUG+ to the logger itself
    but only INFO+ to the console, with a timestamped format.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if this is called more than once
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler()      # defaults to stderr
        console.setLevel(logging.INFO)

        fmt = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console.setFormatter(fmt)

        logger.addHandler(console)

    return logger
