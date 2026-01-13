import logging
import sys
from typing import Any

from pydantic import Field, model_validator

from lh_v2.util import (
    DEFAULT_LOG_LEVEL,
    BaseParamsModel,
    CustomLoggingLevels,
    add_custom_levels,
    silence_loud_loggers,
)


class BaseLoggerParamsModel(BaseParamsModel):
    """
    Base model for logger parameters with string coercion for logging levels.

    This class provides validation and coercion of logging level strings to their
    corresponding integer values from either CustomLoggingLevels or the standard
    logging module.

    Attributes
    ----------
    Inherits all attributes from BaseParamsModel.

    Methods
    -------
    _coerce_strings(data)
        Validates and coerces string logging levels to their integer values.

    Notes
    -----
    The validator attempts to match logging level strings in the following order:
    1. CustomLoggingLevels enumeration members
    2. Standard logging module levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    3. Falls back to DEFAULT_LOG_LEVEL() if no match is found

    Examples
    --------
    >>> params = BaseLoggerParamsModel(root="INFO", custom="DEBUG")
    >>> # "INFO" and "DEBUG" are automatically converted to their integer values
    """

    @model_validator(mode='before')
    @classmethod
    def _coerce_strings(cls, data: Any) -> Any:
        if isinstance(data, dict) and len(data) > 0:
            for key in data.keys():
                level = data[key]
                if isinstance(level, str):
                    if level.upper() in CustomLoggingLevels.__members__:
                        data[key] = CustomLoggingLevels[level.upper()].value
                    else:
                        try:
                            data[key] = getattr(logging, level.upper())
                        except Exception:
                            data[key] = DEFAULT_LOG_LEVEL()
        return data


class IOLoggingParams(BaseLoggerParamsModel):
    """
    Parameters for configuring input/output logging levels.

    This class defines logging levels for different I/O operations within the application,
    allowing fine-grained control over logging verbosity for general operations and
    data loading processes.

    Attributes
    ----------
    general : int
        Logging level for general I/O operations. Defaults to the system's default log level.
    data_loading : int
        Logging level specifically for data loading operations. Defaults to the system's default log level.

    See Also
    --------
    BaseLoggerParamsModel : Base class for logger parameter models.
    DEFAULT_LOG_LEVEL : Function that returns the default logging level.

    Examples
    --------
    >>> params = IOLoggingParams()
    >>> params.general = logging.INFO
    >>> params.data_loading = logging.DEBUG
    """

    general: int = DEFAULT_LOG_LEVEL()
    data_loading: int = DEFAULT_LOG_LEVEL()


class DriverAnalysisLoggingParams(BaseLoggerParamsModel):
    """
    Logging configuration parameters for driver analysis operations.

    This class defines logging levels for different components of the driver
    analysis pipeline, allowing fine-grained control over log verbosity.

    Attributes
    ----------
    general : int
        Logging level for general driver analysis operations.
        Defaults to the system default log level.
    lag_handling : int
        Logging level for lag handling operations during driver analysis.
        Defaults to the system default log level.
    driver_ranking : int
        Logging level for driver ranking computations.
        Defaults to the system default log level.
    collinearity_removal : int
        Logging level for collinearity removal procedures.
        Defaults to the system default log level.

    See Also
    --------
    BaseLoggerParamsModel : Parent class for logging parameter models.

    Notes
    -----
    All logging levels use the standard Python logging constants (e.g., DEBUG=10,
    INFO=20, WARNING=30, ERROR=40, CRITICAL=50).
    """

    general: int = DEFAULT_LOG_LEVEL()
    lag_handling: int = DEFAULT_LOG_LEVEL()
    driver_ranking: int = DEFAULT_LOG_LEVEL()
    collinearity_removal: int = DEFAULT_LOG_LEVEL()


class ForecastingLoggingParams(BaseLoggerParamsModel):
    """
    Configuration parameters for logging levels in forecasting operations.

    This class defines the logging level settings for different components
    of the forecasting system, allowing granular control over log verbosity
    across various forecasting operations.

    Parameters
    ----------
    general : int, optional
        The default logging level for general operations.
        Default is determined by DEFAULT_LOG_LEVEL().
    model_training : int, optional
        The logging level for model training operations.
        Default is determined by DEFAULT_LOG_LEVEL().
    driver_forecasting : int, optional
        The logging level for driver forecasting operations.
        Default is determined by DEFAULT_LOG_LEVEL().
    account_forecasting : int, optional
        The logging level for account forecasting operations.
        Default is determined by DEFAULT_LOG_LEVEL().

    Notes
    -----
    Logging levels typically follow the standard Python logging module conventions:
    - DEBUG (10): Detailed information for diagnosing problems
    - INFO (20): Confirmation that things are working as expected
    - WARNING (30): Indication of unexpected events
    - ERROR (40): Serious problems that prevent some functionality
    - CRITICAL (50): System-critical errors

    Examples
    --------
    >>> params = ForecastingLoggingParams(
    ...     general=20,
    ...     model_training=10,
    ...     driver_forecasting=20,
    ...     account_forecasting=30
    ... )
    """

    general: int = DEFAULT_LOG_LEVEL()
    driver_forecasting: int = DEFAULT_LOG_LEVEL()
    model_training: int = DEFAULT_LOG_LEVEL()
    hyperparam_opt: int = DEFAULT_LOG_LEVEL()
    account_forecasting: int = DEFAULT_LOG_LEVEL()


class LoggingParams(BaseLoggerParamsModel):
    """
    Configuration parameters for logging across different modules.

    This class manages logging levels and configurations for various components
    of the application, including general operations, parameters, I/O operations,
    driver analysis, and forecasting.

    Attributes
    ----------
    general : int
        Logging level for general application operations. Defaults to DEFAULT_LOG_LEVEL().
    params : int
        Logging level for parameter-related operations. Defaults to DEFAULT_LOG_LEVEL().
    io : IOLoggingParams
        Logging configuration for input/output operations.
    driver_analysis : DriverAnalysisLoggingParams
        Logging configuration for driver analysis operations.
    forecasting : ForecastingLoggingParams
        Logging configuration for forecasting operations.

    Notes
    -----
    This class inherits from BaseLoggerParamsModel and uses Pydantic's Field
    factory pattern for nested configuration objects.
    """

    general: int = DEFAULT_LOG_LEVEL()
    params: int = DEFAULT_LOG_LEVEL()
    io: IOLoggingParams = Field(default_factory=IOLoggingParams)
    driver_analysis: DriverAnalysisLoggingParams = Field(
        default_factory=DriverAnalysisLoggingParams
    )
    forecasting: ForecastingLoggingParams = Field(
        default_factory=ForecastingLoggingParams
    )


def _setup_io_logging(logging_params: IOLoggingParams):
    """
    Configure logging levels for I/O related loggers.

    Parameters
    ----------
    logging_params : IOLoggingParams
        Object containing logging level configurations for I/O operations,
        including general I/O logging and data loading specific logging.

    Returns
    -------
    None
    """
    logging.getLogger('lh_v2.io').setLevel(level=logging_params.general)
    logging.getLogger('lh_v2.io.data_loading').setLevel(
        level=logging_params.data_loading
    )
    return


def _setup_driver_analysis_logging(logging_params: DriverAnalysisLoggingParams):
    """
    Configure logging levels for driver analysis components.

    Sets up hierarchical logging levels for the driver analysis module and its
    submodules based on the provided configuration parameters.

    Parameters
    ----------
    logging_params : DriverAnalysisLoggingParams
        Configuration object containing logging level settings for different
        driver analysis components including general, lag_handling, driver_ranking,
        and collinearity_removal.

    Returns
    -------
    None

    Notes
    -----
    This function configures logging for the following loggers:
    - 'lh_v2.driver_analysis': General driver analysis logging
    - 'lh_v2.driver_analysis.lagging': Lag handling operations
    - 'lh_v2.driver_analysis.driver_ranking': Driver ranking operations
    - 'lh_v2.driver_analysis.collinearity': Collinearity removal operations
    """
    logging.getLogger('lh_v2.driver_analysis').setLevel(level=logging_params.general)
    logging.getLogger('lh_v2.driver_analysis.lagging').setLevel(
        level=logging_params.lag_handling
    )
    logging.getLogger('lh_v2.driver_analysis.driver_ranking').setLevel(
        level=logging_params.driver_ranking
    )
    logging.getLogger('lh_v2.driver_analysis.collinearity').setLevel(
        level=logging_params.collinearity_removal
    )
    return


def _setup_forecasting_logging(logging_params: ForecastingLoggingParams):
    """
    Configure logging levels for forecasting-related loggers.

    This function sets up the logging configuration for various components of the
    forecasting module by applying the specified logging levels from the provided
    parameters.

    Parameters
    ----------
    logging_params : ForecastingLoggingParams
        A parameter object containing logging level specifications for different
        forecasting components including general, model_training, driver_forecasting,
        and account_forecasting.

    Returns
    -------
    None
        This function modifies logger configurations in place and returns nothing.

    Notes
    -----
    The function configures the following loggers:
    - 'lh_v2.forecasting': General forecasting logger
    - 'lh_v2.forecasting.model_training': Model training specific logger
    - 'lh_v2.forecasting.driver_forecasting': Driver forecasting specific logger
    - 'lh_v2.forecasting.account_forecasting': Account forecasting specific logger
    """
    logging.getLogger('lh_v2.forecasting').setLevel(level=logging_params.general)
    logging.getLogger('lh_v2.forecasting.driver_forecasting').setLevel(
        level=logging_params.driver_forecasting
    )
    logging.getLogger(
        'lh_v2.forecasting.account_forecasting.model_validation'
    ).setLevel(level=logging_params.model_training)
    logging.getLogger(
        'lh_v2.forecasting.account_forecasting.hyperparameter_optimization'
    ).setLevel(level=logging_params.hyperparam_opt)
    logging.getLogger(
        'lh_v2.forecasting.account_forecasting.model_forecasting'
    ).setLevel(level=logging_params.account_forecasting)
    return


def setup_logging(logging_params: LoggingParams):
    """
    Configure the logging system with custom levels and handlers for the application.

    This function sets up the root logger and module-specific loggers with appropriate
    logging levels based on the provided configuration parameters. It adds custom logging
    levels, configures output formatting, and silences verbose third-party loggers.

    Parameters
    ----------
    logging_params : LoggingParams
        Configuration object containing logging level settings for different modules
        including general, params, io, driver_analysis, and forecasting components.

    Returns
    -------
    None

    See Also
    --------
    add_custom_levels : Adds custom logging levels to the logging module
    silence_loud_loggers : Suppresses verbose output from third-party libraries

    Notes
    -----
    The function performs the following setup steps:
    1. Adds custom logging levels via add_custom_levels()
    2. Creates a StreamHandler that outputs to stdout with timestamp formatting
    3. Sets the general logging level from logging_params.general
    4. Configures the root logger to level 0 to capture all messages
    5. Sets module-specific logging levels for params, io, driver_analysis, and forecasting
    6. Silences overly verbose third-party loggers

    Examples
    --------
    >>> from lh_v2.params import LoggingParams
    >>> params = LoggingParams(general=logging.INFO, params=logging.DEBUG)
    >>> setup_logging(params)
    """
    # Register custom logging levels (e.g., TRACE, VERBOSE) with the logging module
    add_custom_levels()

    # Create a handler to output logs to stdout with timestamp formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('%(levelname)s: [%(asctime)s] %(name)s: %(message)s')
    )

    # Configure basic logging with the general log level and the custom handler
    logging.basicConfig(level=logging_params.general, handlers=[handler])

    # Set root logger to level 0 to allow all messages through; filtering happens at handler/logger levels
    root_logger = logging.getLogger()
    root_logger.setLevel(level=0)

    # Configure logging level for the params module
    logging.getLogger('lh_v2.params').setLevel(level=logging_params.params)

    # Configure logging for I/O operations (general and data loading)
    _setup_io_logging(logging_params.io)

    # Configure logging for driver analysis components (lag handling, ranking, collinearity)
    _setup_driver_analysis_logging(logging_params.driver_analysis)

    # Configure logging for forecasting operations (training, driver/account forecasting)
    _setup_forecasting_logging(logging_params.forecasting)

    # Suppress overly verbose logging from third-party libraries
    silence_loud_loggers()

    return
