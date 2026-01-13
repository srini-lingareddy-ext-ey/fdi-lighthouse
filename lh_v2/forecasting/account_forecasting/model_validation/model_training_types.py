import datetime
from dataclasses import dataclass
from typing import Sequence

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params
from lh_v2.io.plotting import plot_account_forecast

from ..account_forecasting_methods import AbstractAccountForecastingMethod
from .model_selection import select_best_forecast_method


@dataclass
class ModelTrainingInput:
    """
    Input data structure for model training configuration.

    Attributes
    ----------
    accounts_drivers : AccountGroupSelectedDrivers
        The selected drivers for each account group used in the model.
    lags : dict[AccountType, dict[DriverName, int]]
        Dictionary mapping account types to their respective driver lags.
        The inner dictionary maps driver names to the number of lag periods.
    classifications : dict[AccountType, dict[DriverName, DriverClassification]]
        Dictionary mapping account types to their driver classifications.
        The inner dictionary maps driver names to their classification types.
    """

    accounts_drivers: dts.AccountGroupSelectedDrivers
    lags: dict[dts.AccountType, dict[dts.DriverName, int]]
    classifications: dict[
        dts.AccountType, dict[dts.DriverName, dts.DriverClassification]
    ]


@dataclass
class ValidatedModels:
    """
    A container class for validated forecasting models and their associated metrics.

    This class manages a collection of trained forecasting models, providing access
    to specific models and their performance metrics through a mapping system.

    Attributes
    ----------
    model_map : dict[aft.AccountForecastingMethodEnum, int]
        A dictionary mapping forecasting method enums to their corresponding indices
        in the trained_models and metrics lists.
    trained_models : list[AbstractAccountForecastingMethod]
        A list of trained forecasting model instances.
    metrics : list[dict[str, float]]
        A list of dictionaries containing performance metrics for each trained model,
        where each dictionary maps metric names to their float values.

    Methods
    -------
    get_trained_model(key)
        Retrieves a trained model by its forecasting method enum key.
    model_metrics(key)
        Retrieves the performance metrics for a model by its forecasting method enum key.

    Examples
    --------
    >>> validated_models = ValidatedModels()
    >>> model = validated_models.get_trained_model(AccountForecastingMethodEnum.LINEAR)
    >>> metrics = validated_models.model_metrics(AccountForecastingMethodEnum.LINEAR)
    """

    model_map: dict[aft.AccountForecastingMethodEnum, int]
    trained_models: list[AbstractAccountForecastingMethod]
    metrics: list[dict[str, float]]

    def get_trained_model(
        self, key: aft.AccountForecastingMethodEnum
    ) -> AbstractAccountForecastingMethod:
        assert key in self.model_map, f'Model {key} not found in validated models.'
        return self.trained_models[self.model_map[key]]

    def model_metrics(self, key: aft.AccountForecastingMethodEnum) -> dict[str, float]:
        assert key in self.model_map, f'Model {key} not found in validated models.'
        return self.metrics[self.model_map[key]]


@dataclass
class ModelTrainingOutput:
    """
    Container for the complete output of the model training process.

    This class holds all artifacts generated during model training, including
    forecasted accounts, performance metrics, and methods for selecting the best
    forecasting method for each account type.

    Attributes
    ----------
    account_map : dict[AccountType, int]
        Dictionary mapping account types to their corresponding indices in the
        forecasted_accounts and metrics lists.
    method_map : dict[AccountForecastingMethodEnum, int]
        Dictionary mapping forecasting method enums to their corresponding indices
        in the nested lists of forecasted accounts and metrics.
    forecasted_accounts : list[list[AccountInfo]]
        Nested list structure where the outer list corresponds to account types
        and the inner list corresponds to different forecasting methods. Contains
        the forecasted account information for each account-method combination.
    metrics : list[list[dict[AccountValidationMetricEnum, float]]]
        Nested list structure containing validation metrics for each account type
        and forecasting method combination. Each dictionary maps metric enums to
        their computed float values.

    Methods
    -------
    select_forecast_methods(selection_metric=AccountValidationMetricEnum.RMSE_PERCENTAGE)
        Selects the best forecasting method for each account type based on the
        specified validation metric.
    plot_forecast(account, method, forecast_daterange)
        Plots the forecast for a specific account using a specified forecasting
        method over the given date range.

    Examples
    --------
    >>> output = ModelTrainingOutput(...)
    >>> best_methods = output.select_forecast_methods()
    >>> output.plot_forecast(account_type, method, (start_date, end_date))
    """

    account_map: dict[dts.AccountType, int]
    method_map: dict[dts.AccountType, dict[aft.AccountForecastingMethodEnum, int]]
    forecasted_accounts: list[list[dts.AccountInfo]]
    metrics: list[list[dict[aft.AccountValidationMetricEnum, float]]]
    best_params: list[params.AccountForecastParams]

    def select_forecast_methods(
        self,
        selection_metric: aft.AccountValidationMetricEnum = aft.AccountValidationMetricEnum.RMSE_PERCENTAGE,
    ) -> dict[dts.AccountType, aft.AccountForecastingMethodEnum]:
        selected_methods: dict[dts.AccountType, aft.AccountForecastingMethodEnum] = {}
        for account_idx, account in enumerate(self.account_map.keys()):
            best_method = select_best_forecast_method(
                method_map=self.method_map[account],
                metrics=self.metrics[account_idx],
                selection_metric=selection_metric,
            )
            selected_methods[account] = best_method

        return selected_methods

    def get_llm_forecast_metrics(
        self,
        llm_metrics: Sequence[aft.AccountValidationMetricEnum],
    ) -> dict[
        dts.AccountType,
        dict[
            aft.AccountForecastingMethodEnum,
            dict[aft.AccountValidationMetricEnum, float],
        ],
    ]:
        llm_forecast_metrics: dict[
            dts.AccountType,
            dict[
                aft.AccountForecastingMethodEnum,
                dict[aft.AccountValidationMetricEnum, float],
            ],
        ] = {}
        for account in self.account_map.keys():
            llm_forecast_metrics[account] = {}
            account_idx = self.account_map[account]
            for method in self.method_map[account].keys():
                llm_forecast_metrics[account][method] = {}
                method_idx = self.method_map[account][method]
                for metric in llm_metrics:
                    llm_forecast_metrics[account][method][metric] = self.metrics[
                        account_idx
                    ][method_idx][metric]

        return llm_forecast_metrics

    def format_best_params(self) -> dict[dts.AccountType, params.AccountForecastParams]:
        best_params_formatted: dict[dts.AccountType, params.AccountForecastParams] = {}
        for account, account_idx in self.account_map.items():
            best_params_formatted[account] = self.best_params[account_idx]
        return best_params_formatted

    def plot_forecast(
        self,
        account: dts.AccountType,
        method: aft.AccountForecastingMethodEnum,
        forecast_daterange: tuple[datetime.date, datetime.date],
    ):
        forecasted_account = self.forecasted_accounts[self.account_map[account]][
            self.method_map[account][method]
        ]

        plot_account_forecast(
            acc=forecasted_account, forecast_daterange=forecast_daterange
        )
        return
