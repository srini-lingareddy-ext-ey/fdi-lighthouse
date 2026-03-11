"""
Docstring for tests.integration.conftest

Integration test fixtures for Lighthouse project.
"""

import pathlib as pth
import warnings

import pytest
import sklearn.exceptions

from lh_v2.account_reconciliation import apply_account_reconciliation
from lh_v2.datatypes import AccountGroupClassifiedDriverGroups
from lh_v2.datatypes.account_reconciliation_types import AccountReconciliationMethodEnum
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.driver_analysis.driver_analysis_types import (
    DriverAnalysisInput,
    DriverAnalysisOutput,
)
from lh_v2.forecasting import create_account_forecasts, train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingInput,
    ModelForecastingOutput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
    ModelTrainingOutput,
)
from lh_v2.params import AccountReconciliationParams, LighthouseParams
from lh_v2.params.account_reconciliation_params.algebraic_formula_parsing import (
    parse_formula,
)
from lh_v2.shared import BASE_NP_DTYPE
from tests.util import read_test_acc_data, read_test_driver_data


@pytest.fixture(scope='session')
def sample_acc_classified_drivers_info(
    sample_params: LighthouseParams,
    testing_acc_data_pth: pth.Path,
    testing_driver_data_pth: pth.Path,
) -> AccountGroupClassifiedDriverGroups:
    """Return a sample AccountGroupClassifiedDriverGroups for testing."""
    acc_group = read_test_acc_data(
        file_path=testing_acc_data_pth,
        accounts=sample_params.accounts,
        product=sample_params.segment,
        location=sample_params.region,
        np_dtype=BASE_NP_DTYPE,
    )

    classified_drivers = read_test_driver_data(
        file_path=testing_driver_data_pth,
        np_dtype=BASE_NP_DTYPE,
    )

    return AccountGroupClassifiedDriverGroups(
        accounts=acc_group,
        classified_drivers=classified_drivers,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture(scope='session')
def sample_analyze_drivers_full(
    sample_acc_classified_drivers_info: AccountGroupClassifiedDriverGroups,
    sample_params: LighthouseParams,
) -> DriverAnalysisOutput:
    """Fixture to run full driver analysis for testing."""
    da_input = DriverAnalysisInput(
        accounts=sample_acc_classified_drivers_info.accounts,
        classified_drivers=sample_acc_classified_drivers_info.classified_drivers,
        np_dtype=sample_acc_classified_drivers_info.np_dtype,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=sklearn.exceptions.ConvergenceWarning,
        )
        analysis_results = analyze_drivers_full(
            accounts_drivers_info=da_input,
            general_params=sample_params.general_params,
            da_params=sample_params.driver_analysis_params,
            output_params=sample_params.output_params,
        )

    return analysis_results


@pytest.fixture(scope='session')
def sample_model_training_validation(
    sample_acc_classified_drivers_info: AccountGroupClassifiedDriverGroups,
    sample_params: LighthouseParams,
    sample_analyze_drivers_full: DriverAnalysisOutput,
) -> ModelTrainingOutput:
    """Fixture to run model training and validation for testing."""
    training_input = ModelTrainingInput(
        accounts_drivers=sample_acc_classified_drivers_info.select_drivers(
            sample_analyze_drivers_full.selected_drivers
        ),
        lags=sample_analyze_drivers_full.format_lags(),
        classifications=sample_analyze_drivers_full.format_classifications(),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=sklearn.exceptions.ConvergenceWarning,
        )
        training_results = train_and_validate_models(
            model_training_info=training_input,
            general_params=sample_params.general_params,
            af_params=sample_params.account_forecast_params,
            df_params=sample_params.driver_forecast_params,
            output_params=sample_params.output_params,
        )

    return training_results


@pytest.fixture(scope='session')
def sample_model_forecasting(
    sample_acc_classified_drivers_info: AccountGroupClassifiedDriverGroups,
    sample_params: LighthouseParams,
    sample_analyze_drivers_full: DriverAnalysisOutput,
    sample_model_training_validation: ModelTrainingOutput,
) -> ModelForecastingOutput:
    """Fixture to run model forecasting for testing."""
    forecasting_input = ModelForecastingInput(
        accounts_drivers=sample_acc_classified_drivers_info.select_drivers(
            sample_analyze_drivers_full.selected_drivers
        ),
        lags=sample_analyze_drivers_full.format_lags(),
        classifications=sample_analyze_drivers_full.format_classifications(),
        selected_model=sample_model_training_validation.select_forecast_methods(),
        best_params=sample_model_training_validation.format_best_params(),
        validation_errors=sample_model_training_validation.get_selected_methods_errors(
            selected_methods=sample_model_training_validation.select_forecast_methods(),
            actuals=sample_acc_classified_drivers_info.accounts,
            val_date_range=(
                sample_params.general_params.validation_start_date,
                sample_params.general_params.validation_end_date,
            ),
        ),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=sklearn.exceptions.ConvergenceWarning,
        )
        forecasting_results = create_account_forecasts(
            forecasting_input=forecasting_input,
            general_params=sample_params.general_params,
            df_params=sample_params.driver_forecast_params,
            output_params=sample_params.output_params,
        )

    return forecasting_results


@pytest.fixture(scope='session')
def sample_account_reconciliation_params_algebraic_formula(
    sample_params: LighthouseParams,
) -> AccountReconciliationParams:
    """Return sample AccountReconciliationParams for testing."""
    new_params = sample_params.account_reconciliation_params.model_copy(deep=True)
    new_params.b_reconcile = True
    new_params.b_allow_default_algebraic_formulas = True
    new_params.method = AccountReconciliationMethodEnum.ALGEBRAIC_FORMULA
    formulas = [
        'gross_margin_c = net_revenue - cogs_total - t_w_total',
        'total_costs_c = cogs_total + t_w_total',
        'price_c = net_revenue / volume',
        'unit_cost_c = cogs_total / volume',
    ]
    parsed_formulas = []
    for formula in formulas:
        parsed_formulas.append(parse_formula(formula))
    new_params.formulas = parsed_formulas
    return new_params


@pytest.fixture(scope='session')
def sample_account_reconciliation_params_mint_adjustment(
    sample_params: LighthouseParams,
) -> AccountReconciliationParams:
    """Return sample AccountReconciliationParams for testing."""
    new_params = sample_params.account_reconciliation_params.model_copy(deep=True)
    new_params.b_reconcile = True
    new_params.b_allow_default_algebraic_formulas = True
    new_params.method = AccountReconciliationMethodEnum.MINT_ADJUSTMENT
    formulas = [
        'gross_margin = net_revenue - cogs_total - t_w_total',
        'total_costs = cogs_total + t_w_total',
    ]
    parsed_formulas = []
    for formula in formulas:
        parsed_formulas.append(parse_formula(formula))
    new_params.formulas = parsed_formulas
    return new_params


@pytest.fixture(scope='session')
def sample_account_reconciliation_algebraic_formula(
    sample_model_forecasting: ModelForecastingOutput,
    sample_account_reconciliation_params_algebraic_formula: AccountReconciliationParams,
    sample_params: LighthouseParams,
) -> ModelForecastingOutput:
    """Fixture to run account reconciliation for testing."""
    return apply_account_reconciliation(
        forecasting_data=sample_model_forecasting,
        reconciliation_params=sample_account_reconciliation_params_algebraic_formula,
        output_params=sample_params.output_params,
    )


@pytest.fixture(scope='session')
def sample_account_reconciliation_mint_adjustment(
    sample_model_forecasting: ModelForecastingOutput,
    sample_account_reconciliation_params_mint_adjustment: AccountReconciliationParams,
    sample_params: LighthouseParams,
) -> ModelForecastingOutput:
    """Fixture to run account reconciliation for testing."""
    return apply_account_reconciliation(
        forecasting_data=sample_model_forecasting,
        reconciliation_params=sample_account_reconciliation_params_mint_adjustment,
        output_params=sample_params.output_params,
    )
