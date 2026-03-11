from enum import Enum


class SPDriverForecastingMethodEnum(str, Enum):
    """
    Enumeration for the different methods used in Scenario Planning
    Driver Perturbation.
    """

    VAR_BASED = 'var_based'


class SPDriverScenarioEnum(str, Enum):
    """
    Enumeration for the different scenarios used in SP Driver Forecasting.
    """

    VERY_LOW = 'very_low'
    LOW = 'low'
    AVG = 'avg'
    HIGH = 'high'
    VERY_HIGH = 'very_high'


class SPExtremaEstimationMethodEnum(str, Enum):
    AUTO = 'auto'
    """
    Automatically select the estimation method based on the account
    forecasting model: correlation for linear models, sampling for
    nonlinear/tree-based models.
    """
    CORRELATION = 'correlation'
    """
    Estimate the extrema of the scenarios by looking at the 
    correlation between the drivers and the target variable.
    """
    EXACT = 'exact'
    """
    Estimate the extrema of the scenarios by checking every 
    possible combination of driver scenarios.
    """
    SAMPLING = 'sampling'
    """
    Estimate the extrema by evaluating a random sample of driver
    scenario combinations and selecting the ones that produce the
    highest/lowest predicted output. More robust than correlation
    for nonlinear models.
    """


class SPExtremaCase(str, Enum):
    MAX = 'max'
    MIN = 'min'
