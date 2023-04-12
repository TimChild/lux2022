import logging

# Custom logging levels
FUNCTION_CALL = 25
logging.addLevelName(FUNCTION_CALL, "FUNCTION")


def function_call(self, message, *args, **kwargs):
    if self.isEnabledFor(FUNCTION_CALL):
        self._log(FUNCTION_CALL, message, args, **kwargs)


logging.Logger.function_call = function_call


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
base_logger.setLevel(logging.DEBUG)

# Create handlers for debug, info, and other levels
debug_handler = logging.StreamHandler()
info_handler = logging.StreamHandler()
other_handler = logging.StreamHandler()

# Set level and add filters for each handler
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(LevelFilter(logging.DEBUG))

info_handler.setLevel(logging.INFO)
info_handler.addFilter(LevelFilter(logging.INFO))

other_handler.setLevel(logging.WARNING)
other_handler.addFilter(LevelAboveFilter(logging.INFO))


# Create formatters and add them to handlers
# info_format = logging.Formatter(
#     '\t%(levelname)-8.8s:%(module)-10.10s.%(name)-10.10s:%(lineno)-4d - %(message)s'
# )
#
# debug_format = logging.Formatter(
#     '\t\t%(levelname)-8.8s:%(module)-10.10s.%(name)-10.10s:%(lineno)-4d - %(message)s'
# )
# other_format = logging.Formatter(
#     '%(levelname)-8.8s:%(module)-10.10s.%(name)-10.10s:%(lineno)-4d - %(message)s'
# )
other_format = logging.Formatter(
    '%(levelname)s:%(module)s.%(name)s:%(lineno)d: %(message)s'
)
info_format = logging.Formatter(
    '\t%(levelname)s:%(module)s.%(name)s:%(lineno)d: %(message)s'
)

debug_format = logging.Formatter(
    '\t\t%(levelname)s:%(module)s.%(name)s:%(lineno)d: %(message)s'
)

debug_handler.setFormatter(debug_format)
info_handler.setFormatter(info_format)
other_handler.setFormatter(other_format)

# Add the handlers to the logger
base_logger.addHandler(debug_handler)
base_logger.addHandler(info_handler)
base_logger.addHandler(other_handler)


# logging.basicConfig(level=logging.INFO, filename='log.log')
base_logger.warning('================== Starting Log =====================')

ALL_LOGGERS = {}


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
    # root = logging.getLogger()
    # if root.handlers:
    #     for handler in root.handlers:
    #         root.removeHandler(handler)
    # logging.basicConfig(
    #     format="%(levelname)-5.5s:%(module)-10.10s.%(name)-10.10s:%(lineno)-4d:%(funcName)-20.20s: %(message)s",
    #     level=level,
    # )
    base_logger.setLevel(level)
    if all_loggers:
        for k, logger in ALL_LOGGERS.items():
            logger.setLevel(base_logger.level)
    base_logger.warning(f'========= Logging level updated to {level} =========')
