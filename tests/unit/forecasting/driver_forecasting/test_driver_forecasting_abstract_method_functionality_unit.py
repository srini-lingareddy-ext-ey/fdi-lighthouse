import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from lh_v2.datatypes import Driver
from lh_v2.forecasting.driver_forecasting.driver_forecasting_methods import (
    LinearRegressionDriverForecastingMethod,
)
from lh_v2.params import LighthouseParams
from lh_v2.util import month_dif


@pytest.mark.unit
@pytest.mark.driver_forecasting
class TestDriverForecastingAbstractMethodFunctionalityUnit:
    """Unit tests for Driver Forecasting Abstract Method Functionality."""

    def test_abstract_class_functionality_lag_0(
        self,
        driver1_info: Driver,
        example_forecasting_params: LighthouseParams,
    ) -> None:
        # Test case: No lag (n_lag=0) means forecaster needs to predict all validation dates
        forecaster = LinearRegressionDriverForecastingMethod(
            driver_info=driver1_info,
            method_params=example_forecasting_params.driver_forecast_params.linear_regression_params,
            training_date=example_forecasting_params.general_params.get_training_daterange(),
            forecast_daterange=example_forecasting_params.general_params.get_validation_daterange(),
            n_lag=0,
        )

        # Verify that all initialization parameters are correctly set
        assert forecaster.driver_info == driver1_info
        assert (
            forecaster.method_params
            == example_forecasting_params.driver_forecast_params.linear_regression_params
        )
        assert (
            forecaster.training_date
            == example_forecasting_params.general_params.get_training_daterange()
        )
        assert (
            forecaster.forecast_daterange
            == example_forecasting_params.general_params.get_validation_daterange()
        )
        assert forecaster.n_lag == 0

        # With no lag, all validation dates require forecasting
        assert forecaster.need_forecast() is True
        # Should return all months in the validation period
        assert forecaster.forecast_dates_required() == [
            example_forecasting_params.general_params.validation_start_date
            + relativedelta(months=i)
            for i in range(
                month_dif(
                    example_forecasting_params.general_params.validation_start_date,
                    example_forecasting_params.general_params.validation_end_date,
                )
                + 1
            )
        ]

        # Extract training data and verify it matches the expected date range
        training_data = forecaster.get_training_data()
        np.allclose(
            training_data,
            driver1_info.apply_daterange(
                start_date=example_forecasting_params.general_params.training_start_date,
                end_date=example_forecasting_params.general_params.training_end_date,
            ).arr,
        )

        # Train the model and ensure it was successfully created
        forecaster.train()
        assert forecaster.model is not None

        # Generate the forecasted driver containing both training and forecast periods
        forecasted_driver = forecaster.get_forecasted_driver(max_lag=0)

        # Verify training period data remains unchanged in the forecasted driver
        assert np.allclose(
            forecasted_driver.apply_daterange(
                start_date=example_forecasting_params.general_params.training_start_date,
                end_date=example_forecasting_params.general_params.training_end_date,
            ).arr,
            training_data,
        )
        assert example_forecasting_params.general_params.training_start_date == min(
            forecasted_driver.dates.keys()
        )
        assert example_forecasting_params.general_params.validation_end_date == max(
            forecasted_driver.dates.keys()
        )

        return

    def test_abstract_class_functionality_lag_less_than_forecast_window(
        self,
        driver1_info: Driver,
        example_forecasting_params: LighthouseParams,
    ) -> None:
        # Test case: Lag of 2 months is less than validation window
        # Some validation dates can use lagged actual data, rest need forecasting
        lag = 2
        forecaster = LinearRegressionDriverForecastingMethod(
            driver_info=driver1_info,
            method_params=example_forecasting_params.driver_forecast_params.linear_regression_params,
            training_date=example_forecasting_params.general_params.get_training_daterange(),
            forecast_daterange=example_forecasting_params.general_params.get_validation_daterange(),
            n_lag=lag,
        )

        # Verify lag is correctly set
        assert forecaster.n_lag == lag

        # With lag < validation window, forecasting is still needed but for fewer dates
        assert forecaster.need_forecast() is True
        # Only need to forecast validation dates minus the lag period
        assert forecaster.forecast_dates_required() == [
            example_forecasting_params.general_params.validation_start_date
            + relativedelta(months=i)
            for i in range(
                month_dif(
                    example_forecasting_params.general_params.validation_start_date,
                    example_forecasting_params.general_params.validation_end_date,
                )
                + 1
                - lag,
            )
        ]

        # Training data extraction is independent of lag
        training_data = forecaster.get_training_data()
        np.allclose(
            training_data,
            driver1_info.apply_daterange(
                start_date=example_forecasting_params.general_params.training_start_date,
                end_date=example_forecasting_params.general_params.training_end_date,
            ).arr,
        )

        # Train the model since forecasting is needed
        forecaster.train()
        assert forecaster.model is not None

        # Generate forecasted driver accounting for the lag
        forecasted_driver = forecaster.get_forecasted_driver(max_lag=lag)

        # Training data should remain unchanged in forecasted driver
        assert np.allclose(
            forecasted_driver.apply_daterange(
                start_date=example_forecasting_params.general_params.training_start_date,
                end_date=example_forecasting_params.general_params.training_end_date,
            ).arr,
            training_data,
        )
        assert example_forecasting_params.general_params.training_start_date == min(
            forecasted_driver.dates.keys()
        )
        # Forecasted driver ends at validation end minus the lag
        assert (
            example_forecasting_params.general_params.validation_end_date
            - relativedelta(months=lag)
            == max(forecasted_driver.dates.keys())
        )

        return

    def test_abstract_class_functionality_lag_greater_than_forecast_window(
        self,
        driver1_info: Driver,
        example_forecasting_params: LighthouseParams,
    ) -> None:
        # Test case: Lag of 12 months exceeds validation window
        # All validation dates can use lagged historical data, no forecasting needed
        lag = 12
        forecaster = LinearRegressionDriverForecastingMethod(
            driver_info=driver1_info,
            method_params=example_forecasting_params.driver_forecast_params.linear_regression_params,
            training_date=example_forecasting_params.general_params.get_training_daterange(),
            forecast_daterange=example_forecasting_params.general_params.get_validation_daterange(),
            n_lag=lag,
        )

        # Verify lag is correctly set
        assert forecaster.n_lag == lag

        # With lag > validation window, no forecasting is required
        assert forecaster.need_forecast() is False
        # No forecast dates are needed
        assert forecaster.forecast_dates_required() == []

        # Training data can still be extracted even though forecasting isn't needed
        training_data = forecaster.get_training_data()
        np.allclose(
            training_data,
            driver1_info.apply_daterange(
                start_date=example_forecasting_params.general_params.training_start_date,
                end_date=example_forecasting_params.general_params.training_end_date,
            ).arr,
        )

        # No model should be trained since forecasting is not required
        forecaster.train()
        assert forecaster.model is None

        # Get forecasted driver which only contains historical data due to large lag
        forecasted_driver = forecaster.get_forecasted_driver(max_lag=lag)

        assert example_forecasting_params.general_params.training_start_date == min(
            forecasted_driver.dates.keys()
        )
        # Forecasted driver ends at validation end minus the large lag
        assert (
            example_forecasting_params.general_params.validation_end_date
            - relativedelta(months=lag)
            == max(forecasted_driver.dates.keys())
        )

        return
