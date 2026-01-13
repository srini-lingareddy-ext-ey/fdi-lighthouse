import logging
from enum import Enum
from typing import Any, Protocol, cast


def DEFAULT_LOG_LEVEL() -> int:
    return logging.WARNING


class CustomLoggingLevels(int, Enum):
    TIMING = 15


class CustomLogger(Protocol):
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def timing(self, message: str, *args: Any, **kwargs: Any) -> None: ...


def _add_custom_level_to_logger(level: int, level_name: str):
    logging.addLevelName(level=level, levelName=level_name)

    def log_for_level(
        self: logging.Logger, message: str, *args: Any, **kwargs: Any
    ) -> None:
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    setattr(logging.Logger, level_name.lower(), log_for_level)
    return


def add_custom_levels():
    for level in CustomLoggingLevels:
        logging.addLevelName(level=level.value, levelName=level.name)
        _add_custom_level_to_logger(level=level.value, level_name=level.name)
    return


def get_logger(name: str) -> CustomLogger:
    return cast(CustomLogger, logging.getLogger(name))


def configure_optuna_logging(level: int = logging.ERROR) -> None:
    logger = logging.getLogger('optuna')

    # Remove any handlers Optuna attached (or will attach before you call this)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Let your root/app handlers format + route Optuna logs
    logger.propagate = True
    logger.setLevel(level)


def silence_loud_loggers() -> None:
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('pandas').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    configure_optuna_logging(level=logging.ERROR)
    """optuna_logger = logging.getLogger('optuna')
    optuna_logger.handlers.clear()
    optuna_logger.propagate = True
    optuna_logger.setLevel(logging.ERROR)"""
    return
