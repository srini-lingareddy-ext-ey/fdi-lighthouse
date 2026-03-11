import datetime
from abc import ABC, abstractmethod
from typing import Any, Optional

from dateutil.relativedelta import relativedelta

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.stats as stats
from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountValidationMetricEnum,
)
from lh_v2.params.forecasting_params import BaseAccountForecastParams
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif


def _std_metric_func(model: ArrayF, true: ArrayF) -> float:
    return stats.std_slr(model)


VALIDATION_METRIC_MAP = {
    AccountValidationMetricEnum.MSE_PERCENTAGE: stats.mse_percentage,
    AccountValidationMetricEnum.RMSE_PERCENTAGE: stats.rmse_percentage,
    AccountValidationMetricEnum.MAPE: stats.mape,
    AccountValidationMetricEnum.STD: _std_metric_func,
}


class AbstractAccountForecastingMethod(ABC):
    """
    Abstract base class for account forecasting methods.

    This class provides the interface for implementing various account forecasting
    methods. Subclasses must implement the abstract methods to define specific
    forecasting algorithms.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        The account driver information used for training the forecasting model.
    best_lags : dict[dts.DriverName, int]
        Dictionary mapping driver names to their optimal lag values.
    training_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the training period.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the forecasting period.
    model_params : BaseAccountForecastParams
        Parameters for configuring the forecasting model.

    Attributes
    ----------
    info : dts.AccountDriverGroup
        Stored account driver information.
    best_lags : dict[dts.DriverName, int]
        Stored optimal lag values for each driver.
    training_daterange : tuple[datetime.date, datetime.date]
        Stored training date range.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Stored forecasting date range.
    model_params : BaseAccountForecastParams
        Stored model parameters.
    model : Optional[Any]
        The trained forecasting model. Initially set to None until training occurs.

    Methods
    -------
    name()
        Returns the name identifier of the forecasting method.
    get_forecasting_dates()
        Returns list of dates for the forecasting period.
    get_training_data()
        Returns the training data subset from account driver information.
    get_forecasting_input_data()
        Returns the input data needed for generating forecasts.
    get_validation_data()
        Returns the validation data for model evaluation.
    train()
        Trains the forecasting model using the provided account driver information.
    apply()
        Apply the trained model to generate forecasts.
    forecast()
        Generate forecast and return as AccountInfo object.
    validate()
        Validates the trained model against validation data and returns performance metrics.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly. Subclasses
    must implement the abstract methods: `name()`, `train()`, and `apply()`.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: BaseAccountForecastParams,
    ):
        self.info = info
        self.best_lags = best_lags
        self.training_daterange = training_daterange
        self.forecast_daterange = forecast_daterange
        self.model_params = model_params
        self.training_info: Optional[dts.AccountDriverGroup] = None
        self.forecasting_input_info: Optional[dts.AccountDriverGroup] = None
        self.validation_info: Optional[dts.AccountInfo] = None
        self.model: Optional[Any] = None
        return

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def is_linear() -> bool:
        raise NotImplementedError()

    def set_model_params(self, model_params: BaseAccountForecastParams) -> None:
        assert type(model_params) is type(self.model_params), (
            'model_params must be of the same type as the existing model_params.'
        )
        self.model_params = model_params
        return

    def get_forecasting_dates(self) -> list[datetime.date]:
        """
        Generate a list of monthly dates within the forecast date range.

        This method creates a sequence of dates starting from the first date in the
        forecast date range and incrementing by one month until reaching the end date
        (inclusive).

        Returns
        -------
        list[datetime.date]
            A list of datetime.date objects representing the first day of each month
            within the forecast date range, including both the start and end months.

        Examples
        --------
        >>> # Assuming forecast_daterange is (date(2024, 1, 1), date(2024, 3, 1))
        >>> forecaster.get_forecasting_dates()
        [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), datetime.date(2024, 3, 1)]
        """
        return [
            self.forecast_daterange[0] + relativedelta(months=i)
            for i in range(
                month_dif(
                    start_date=self.forecast_daterange[0],
                    end_date=self.forecast_daterange[1],
                )
                + 1
            )
        ]

    def set_training_data(self) -> None:
        self.training_info = self.info.apply_daterange(
            start_date=self.training_daterange[0],
            end_date=self.training_daterange[1],
            best_lags=self.best_lags,
            b_training=True,
        )
        return

    def get_training_data(self) -> dts.AccountDriverGroup:
        """
        Retrieve training data for the forecasting model.

        This method applies the training date range to the account driver group information
        and returns the filtered data suitable for model training.

        Returns
        -------
        dts.AccountDriverGroup
            The account driver group data filtered to the training date range, with
            best lags applied and configured for training mode.

        Notes
        -----
        The method uses the following instance attributes:
        - self.training_daterange: A tuple/list with start and end dates
        - self.best_lags: The optimal lag values to apply
        - self.info: The account driver group information object

        See Also
        --------
        apply_daterange : Method that performs the actual date filtering
        """
        if self.training_info is None:
            self.set_training_data()

        assert self.training_info is not None
        return self.training_info

    def set_forecasting_input_data(
        self, new_drivers_data: dts.DriverGroup | None = None
    ) -> None:
        if new_drivers_data is None:
            self.forecasting_input_info = dts.AccountDriverGroup(
                account=self.info.account.apply_daterange(
                    start_date=self.training_daterange[0]
                    + relativedelta(months=max(self.best_lags.values())),
                    end_date=self.forecast_daterange[0] - relativedelta(months=1),
                ),
                drivers=self.info.drivers.apply_daterange_lags(
                    start_date=self.training_daterange[0],
                    end_date=self.forecast_daterange[1],
                    lags=self.best_lags,
                    b_training=True,
                ),
                np_dtype=self.info.np_dtype,
            )
            return

        else:
            assert set(new_drivers_data.get_ordered_drivers()) == set(
                self.info.drivers.get_ordered_drivers()
            ), 'new_drivers_data must contain the same drivers as self.info.drivers.'

            self.forecasting_input_info = dts.AccountDriverGroup(
                account=self.info.account.apply_daterange(
                    start_date=self.training_daterange[0]
                    + relativedelta(months=max(self.best_lags.values())),
                    end_date=self.forecast_daterange[0] - relativedelta(months=1),
                ),
                drivers=new_drivers_data.apply_daterange_lags(
                    start_date=self.training_daterange[0],
                    end_date=self.forecast_daterange[1],
                    lags=self.best_lags,
                    b_training=True,
                ),
                np_dtype=self.info.np_dtype,
            )
            return

    def get_forecasting_input_data(self) -> dts.AccountDriverGroup:
        """
        Get the input data required for forecasting.

        This method constructs an AccountDriverGroup containing the account data and
        driver data adjusted for the appropriate date ranges and lags needed for
        forecasting operations.

        Returns
        -------
        dts.AccountDriverGroup
            An AccountDriverGroup instance containing:
            - account: Account data adjusted to start from training_daterange[0] plus
              the maximum lag value, and ending one month before forecast_daterange[0]
            - drivers: Driver data adjusted for date ranges and lags, spanning from
              training_daterange[0] to forecast_daterange[1], with training mode enabled
            - np_dtype: The numpy data type from the info object

        Notes
        -----
        The account date range is shifted forward by the maximum lag value to ensure
        proper alignment with lagged driver data during model training and forecasting.
        """
        if self.forecasting_input_info is None:
            self.set_forecasting_input_data()

        assert self.forecasting_input_info is not None
        return self.forecasting_input_info

    def set_validation_data(self) -> None:
        self.validation_info = self.info.account.apply_daterange(
            start_date=self.forecast_daterange[0],
            end_date=self.forecast_daterange[1],
        )
        return

    def get_validation_data(self) -> dts.AccountInfo:
        """
        Get validation data for the forecast period.

        Returns
        -------
        dts.AccountInfo
            Account information filtered to the forecast date range, containing
            data between the forecast start and end dates.

        Notes
        -----
        This method applies the forecast date range to the account data to extract
        the relevant validation subset for comparing forecast results.
        """
        if self.validation_info is None:
            self.set_validation_data()

        assert self.validation_info is not None
        return self.validation_info

    @abstractmethod
    def train(self) -> None:
        """
        Train the forecasting model.

        This method should be implemented by subclasses to define the specific
        training procedure for the forecasting model, and store the model
        in self.model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        Subclasses should override this method to implement their specific
        training logic, including data preparation, model fitting.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(self) -> ArrayF:
        """
        Apply the forecasting method to generate predictions.

        Returns
        -------
        ArrayF
            Array of forecasted values.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def apply_vectorized(self, arr_input: ArrayF) -> ArrayF:
        """
        Apply the forecasting method to a vectorized input array.
        The array should be of shape (n_forecasts, n_features, n_samples)
        and the output should be of shape (n_forecasts, n_samples).
        """
        raise NotImplementedError()

    def forecast(self) -> dts.AccountInfo:
        """
        Generate forecast values for the account and add them to account information.

        This method combines the forecasting dates and forecast values to create
        a complete forecast for the account.

        Returns
        -------
        dts.AccountInfo
            Account information object updated with forecast dates and values.

        See Also
        --------
        get_forecasting_dates : Retrieves the dates for which forecasts are generated.
        apply : Computes the forecast values for the forecasting period.
        """
        return self.info.account.add_forecast_vals(
            forecast_dates=self.get_forecasting_dates(),
            forecast_values=self.apply(),
        )

    def validate(
        self,
    ) -> tuple[dts.AccountInfo, dict[AccountValidationMetricEnum, float]]:
        """
        Validate the forecast model against actual data and compute performance metrics.

        This method generates forecasts for the validation period, compares them against
        the true validation data, and calculates multiple evaluation metrics to assess
        model performance.

        Returns
        -------
        tuple[dts.AccountInfo, dict[AccountValidationMetricEnum, float]]
            A tuple containing:
            - AccountInfo : The complete forecast object including validation period
            - dict : Dictionary mapping metric enums to their computed values, including:
                * MSE_PERCENTAGE : Mean Squared Error as a percentage
                * RMSE_PERCENTAGE : Root Mean Squared Error as a percentage
                * MAPE : Mean Absolute Percentage Error

        Notes
        -----
        The validation metrics are computed by comparing forecasted values against
        actual values within the forecast_daterange period. Lower metric values
        generally indicate better model performance.

        See Also
        --------
        forecast : Generates the forecast values
        get_validation_data : Retrieves the actual data for validation
        """
        # Generate the complete forecast for the validation period
        forecast = self.forecast()

        # Extract predicted values for the forecast date range
        predictions = forecast.apply_daterange(
            start_date=self.forecast_daterange[0],
            end_date=self.forecast_daterange[1],
        ).arr

        # Get actual values for comparison
        true = self.get_validation_data().arr

        # Initialize dictionary to store validation metrics
        metrics_dict: dict[AccountValidationMetricEnum, float] = {}

        for metric in AccountValidationMetricEnum:
            metrics_dict[metric] = VALIDATION_METRIC_MAP[metric](
                model=predictions, true=true
            )

        # Return both the forecast and computed metrics
        return (
            forecast,
            metrics_dict,
        )
