import logging
from logging.handlers import RotatingFileHandler

FILEPATH = "logfile.log"

ALL_LOGGERS = {}

DEFAULT_LEVEL = logging.WARNING


VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")


def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


logging.Logger.verbose = verbose


class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


class LevelAboveFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno > self.level


# Create a custom logger
base_logger = logging.getLogger(__name__)
base_logger.setLevel(DEFAULT_LEVEL)
base_logger.handlers = []

# Create handlers for debug, info, and other levels
# Clear the logfile each time
with open(FILEPATH, "w") as f:
    pass

verbose_handler = RotatingFileHandler(FILEPATH)
debug_handler = RotatingFileHandler(FILEPATH)
info_handler = RotatingFileHandler(FILEPATH)
other_handler = RotatingFileHandler(FILEPATH)

# Set level and add filters for each handler
verbose_handler.setLevel(VERBOSE)
verbose_handler.addFilter(LevelFilter(VERBOSE))

debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(LevelFilter(logging.DEBUG))

info_handler.setLevel(logging.INFO)
info_handler.addFilter(LevelFilter(logging.INFO))

other_handler.setLevel(logging.WARNING)
other_handler.addFilter(LevelAboveFilter(logging.INFO))

# Create formatters and add them to handlers
FORMAT = "%(levelname)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s"
other_format = logging.Formatter(f"{FORMAT}")
info_format = logging.Formatter(f"\t{FORMAT}")

debug_format = logging.Formatter(f"\t\t{FORMAT}")

verbose_format = logging.Formatter("\t\t\t%(levelname)s: %(message)s")

verbose_handler.setFormatter(verbose_format)
debug_handler.setFormatter(debug_format)
info_handler.setFormatter(info_format)
other_handler.setFormatter(other_format)

# Add the handlers to the logger
base_logger.addHandler(verbose_handler)
base_logger.addHandler(debug_handler)
base_logger.addHandler(info_handler)
base_logger.addHandler(other_handler)


# logging.basicConfig(level=logging.INFO, filename='log.log')
base_logger.warning("================== Starting Log =====================")


def get_logger(name) -> logging.Logger:
    if name not in ALL_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers = base_logger.handlers
        logger.setLevel(base_logger.level)
        # logger.name
        ALL_LOGGERS[name] = logger
    return ALL_LOGGERS[name]


def update_logging_level(level, all_loggers=True):
    """Updates the logging level"""
    base_logger.setLevel(level)
    if all_loggers:
        for k, logger in ALL_LOGGERS.items():
            logger.setLevel(base_logger.level)
    base_logger.warning(f"========= Logging level updated to {level} =========")
