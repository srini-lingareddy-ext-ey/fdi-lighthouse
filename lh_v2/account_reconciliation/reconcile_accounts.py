from lh_v2.datatypes.account_reconciliation_types import AccountReconciliationMethodEnum
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingOutput,
)
from lh_v2.params import AccountReconciliationParams
from lh_v2.util import get_logger

from .reconciliation_methods import (
    apply_algebraic_formulas_direct,
    apply_formulas_mint,
    verify_mint_adjustment_possible,
)

logger = get_logger(__name__)


def apply_account_reconciliation(
    forecasting_data: ModelForecastingOutput,
    reconciliation_params: AccountReconciliationParams,
) -> ModelForecastingOutput:
    """
    Apply account reconciliation to forecasting data using specified method.

    This function reconciles account forecasts by applying formulas to ensure
    accounting equations are satisfied. It supports multiple reconciliation
    methods including algebraic formulas and MINT adjustment.

    Parameters
    ----------
    forecasting_data : ModelForecastingOutput
        The forecasting output data containing account forecasts, date range,
        and validation errors.
    reconciliation_params : AccountReconciliationParams
        Parameters controlling the reconciliation process, including the
        reconciliation method, formulas to apply, and fallback options.

    Returns
    -------
    ModelForecastingOutput
        The reconciled forecasting output with updated account forecasts.
        Returns the original data unchanged if reconciliation is disabled.

    Raises
    ------
    ValueError
        If MINT adjustment reconciliation is not possible and default
        algebraic formulas are not allowed.
    NotImplementedError
        If an unsupported reconciliation method is specified.

    Notes
    -----
    If MINT adjustment reconciliation is not possible and
    `b_allow_default_algebraic_formulas` is True, the method will
    automatically fall back to algebraic formula reconciliation.
    """
    # Check if reconciliation is enabled in the parameters
    if not reconciliation_params.b_reconcile:
        logger.info('Account reconciliation is disabled. Skipping reconciliation step.')
        return forecasting_data

    # For MINT adjustment method, verify prerequisites before proceeding
    if reconciliation_params.method == AccountReconciliationMethodEnum.MINT_ADJUSTMENT:
        # Extract the set of accounts that have been forecasted
        precalced_accounts = set(
            forecasting_data.accounts_forecasts.get_ordered_accounts()
        )

        # Verify that all required accounts for MINT adjustment are available
        if not verify_mint_adjustment_possible(
            reconciliation_params.formulas,
            precalced_accounts,
        ):
            # Attempt fallback to algebraic formulas if allowed
            if reconciliation_params.b_allow_default_algebraic_formulas:
                logger.warning(
                    'MINT adjustment reconciliation is not possible with the provided '
                    'formulas. Falling back to algebraic formula reconciliation.'
                )
                # Switch reconciliation method to algebraic formula
                reconciliation_params.method = (
                    AccountReconciliationMethodEnum.ALGEBRAIC_FORMULA
                )
            else:
                # Raise error if fallback is not allowed
                raise ValueError(
                    'MINT adjustment reconciliation is not possible with the provided '
                    'formulas and default algebraic formulas are not allowed.'
                )

    # Apply the appropriate reconciliation method based on configuration
    match reconciliation_params.method:
        case AccountReconciliationMethodEnum.ALGEBRAIC_FORMULA:
            # Apply algebraic formulas directly to compute dependent accounts
            reconciled_data = apply_algebraic_formulas_direct(
                forecast_data=forecasting_data.accounts_forecasts,
                formulas=reconciliation_params.formulas,
            )

        case AccountReconciliationMethodEnum.MINT_ADJUSTMENT:
            # Apply MINT adjustment method to reconcile accounts
            reconciled_data = apply_formulas_mint(
                accounts_forecasts=forecasting_data.accounts_forecasts,
                formulas=reconciliation_params.formulas,
                validation_errors=forecasting_data.validation_errors,
            )

        case _:
            # Raise error for unsupported reconciliation methods
            raise NotImplementedError(
                f'Reconciliation method {reconciliation_params.method} not implemented.'
            )

    # Return new ModelForecastingOutput with reconciled account data
    return ModelForecastingOutput(
        accounts_forecasts=reconciled_data,
        forecast_daterange=forecasting_data.forecast_daterange,
        validation_errors=forecasting_data.validation_errors,
    )
