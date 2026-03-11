import datetime
from dataclasses import dataclass

from lh_v2.datatypes import (
    AccountGroupInfo,
    AccountGroupSelectedDrivers,
    AccountType,
    DriverClassification,
    DriverName,
)
from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountForecastingMethodEnum,
)
from lh_v2.params import AccountForecastParams
from lh_v2.shared import ArrayF


@dataclass
class ModelForecastingInput:
    """
    Input configuration for model-based forecasting of accounts.

    This class encapsulates all the necessary parameters and configurations
    required to perform forecasting for different account types using various
    driver-based models.

    Attributes
    ----------
    accounts_drivers : AccountGroupSelectedDrivers
        The selected drivers associated with each account group that will be
        used for forecasting calculations.
    lags : dict[AccountType, dict[DriverName, int]]
        A nested dictionary mapping account types to their respective drivers
        and the number of lag periods to apply for each driver.
    classifications : dict[AccountType, dict[DriverName, DriverClassification]]
        A nested dictionary mapping account types to their drivers and the
        classification type for each driver (e.g., leading, lagging, coincident).
    selected_model : dict[
            AccountType,
            tuple[
                AccountForecastingMethodEnum,
                Optional[AbstractAccountForecastingMethod]
            ]
        ]
        A dictionary mapping each account type to a tuple containing the
        forecasting method enumeration and an optional instance of the
        forecasting method implementation.
    best_params : dict[AccountType, AccountForecastParams]
        A dictionary mapping each account type to its best hyperparameters
        determined during model selection or tuning.
    validation_errors : dict[AccountType, ArrayF]
        Point-wise validation errors for each account type's forecast,
        represented as arrays of floating-point numbers.

    Notes
    -----
    This class serves as a data container for forecasting inputs and does not
    implement any forecasting logic itself.
    """

    accounts_drivers: AccountGroupSelectedDrivers
    lags: dict[AccountType, dict[DriverName, int]]
    classifications: dict[AccountType, dict[DriverName, DriverClassification]]
    selected_model: dict[AccountType, AccountForecastingMethodEnum]
    best_params: dict[AccountType, AccountForecastParams]
    validation_errors: dict[AccountType, ArrayF]


@dataclass
class ModelForecastingOutput:
    """
    Container for model forecasting outputs and visualization methods.

    Attributes
    ----------
    accounts_forecasts : AccountGroupInfo
        Collection of account forecasts for different account types.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates defining the forecasting period.
    validation_errors : dict[AccountType, ArrayF]
        Point-wise validation errors for each account type's forecast.

    Methods
    -------
    get_forecasts()
        Returns account forecasts filtered to the forecast date range.
    plot_forecast(account_type)
        Generates a visualization of the forecast for a specific account type.
    """

    accounts_forecasts: AccountGroupInfo
    forecast_daterange: tuple[datetime.date, datetime.date]
    validation_errors: dict[AccountType, ArrayF]

    def get_forecasts(self) -> AccountGroupInfo:
        return self.accounts_forecasts.apply_daterange(
            start_date=self.forecast_daterange[0],
            end_date=self.forecast_daterange[1],
        )
