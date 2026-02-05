import datetime

import numpy as np
import pytest

from lh_v2.datatypes import AccountDriverGroup, AccountInfo, DriverGroup, DriverName
from lh_v2.forecasting.account_forecasting.account_forecasting_methods import (
    LinearRegressionAccountForecastingMethod,
)
from lh_v2.params import LighthouseParams


@pytest.mark.unit
@pytest.mark.account_forecasting
class TestAccountForecastingAbstractMethodFunctionalityUnit:
    """Unit tests for Account Forecasting Abstract Method Functionality."""

    def test_abstract_class_functionality_validation(
        self,
        account1_info: AccountInfo,
        driver_group1: DriverGroup,
        example_forecasting_params: LighthouseParams,
    ):
        # Create an AccountDriverGroup combining account and driver data
        account_driver_info = AccountDriverGroup(
            account=account1_info,
            drivers=driver_group1,
            np_dtype=account1_info.np_dtype,
        )
        # Set all driver lags to 0 (no lag) for this test
        best_lags: dict[DriverName, int] = {
            driver_name: 0 for driver_name in driver_group1.get_ordered_drivers()
        }
        # Initialize the linear regression forecaster with account and driver data
        forecaster = LinearRegressionAccountForecastingMethod(
            info=account_driver_info,
            best_lags=best_lags,
            training_daterange=example_forecasting_params.general_params.get_training_daterange(),
            forecast_daterange=example_forecasting_params.general_params.get_validation_daterange(),
            model_params=example_forecasting_params.account_forecast_params.linear_regression_params,
        )

        # Verify that all initialization parameters are correctly set
        assert forecaster.info == account_driver_info
        assert forecaster.best_lags == best_lags
        assert (
            forecaster.training_daterange
            == example_forecasting_params.general_params.get_training_daterange()
        )
        assert (
            forecaster.forecast_daterange
            == example_forecasting_params.general_params.get_validation_daterange()
        )
        assert (
            forecaster.model_params
            == example_forecasting_params.account_forecast_params.linear_regression_params
        )

        # Verify that account training data matches the expected training period
        assert np.allclose(
            forecaster.get_training_data().account.arr,
            account1_info.apply_daterange(
                example_forecasting_params.general_params.training_start_date,
                example_forecasting_params.general_params.training_end_date,
            ).arr,
        )

        # Create date ranges for each driver in the training period
        start_dates: dict[DriverName, datetime.date] = {
            driver_name: example_forecasting_params.general_params.training_start_date
            for driver_name in driver_group1.get_ordered_drivers()
        }
        end_dates: dict[DriverName, datetime.date] = {
            driver_name: example_forecasting_params.general_params.training_end_date
            for driver_name in driver_group1.get_ordered_drivers()
        }
        # Verify that driver training data matches the expected training period
        assert np.allclose(
            forecaster.get_training_data().drivers.arr,
            driver_group1.apply_daterange(
                start_dates=start_dates,
                end_dates=end_dates,
            ).arr,
        )
