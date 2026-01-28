"""
Docstring for tests.integration.account_reconciliation.test_account_reconciliation

These tests verify the account reconciliation functionality.
"""

import numpy as np
import pytest

from lh_v2.datatypes import AccountType
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingOutput,
)
from lh_v2.params import AccountReconciliationParams


@pytest.mark.integration
@pytest.mark.account_reconciliation
class TestAccountReconciliationIntegration:
    """Integration tests for account reconciliation."""

    def test_account_reconciliation_algebraic_formula(
        self,
        sample_account_reconciliation_algebraic_formula: ModelForecastingOutput,
        sample_account_reconciliation_params_algebraic_formula: AccountReconciliationParams,
    ):
        """Test that account reconciliation runs without errors."""
        assert isinstance(
            sample_account_reconciliation_algebraic_formula, ModelForecastingOutput
        )
        # Further checks can be added here to verify reconciliation results
        new_accounts = []
        for formula in sample_account_reconciliation_params_algebraic_formula.formulas:
            new_accounts.append(formula.lhs)

        for account in new_accounts:
            assert (
                account
                in sample_account_reconciliation_algebraic_formula.accounts_forecasts.get_ordered_accounts()
            )

        # check that gross_margin_c = net_revenue - cogs_total - t_w_total
        assert np.isclose(
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('gross_margin_c')
            ].arr,
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('net_revenue')
            ].arr
            - sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('cogs_total')
            ].arr
            - sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('t_w_total')
            ].arr,
        ).all()
        # check that total_costs_c = cogs_total + t_w_total
        assert np.isclose(
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('total_costs_c')
            ].arr,
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('cogs_total')
            ].arr
            + sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('t_w_total')
            ].arr,
        ).all()
        # check that price_c = net_revenue / volume
        assert np.isclose(
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('price_c')
            ].arr,
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('net_revenue')
            ].arr
            / sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('volume')
            ].arr,
        ).all()
        # check that unit_cost_c = cogs_total / volume
        assert np.isclose(
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('unit_cost_c')
            ].arr,
            sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('cogs_total')
            ].arr
            / sample_account_reconciliation_algebraic_formula.accounts_forecasts[
                AccountType('volume')
            ].arr,
        ).all()
        return

    def test_account_reconciliation_mint_adjustment(
        self,
        sample_account_reconciliation_mint_adjustment: ModelForecastingOutput,
    ):
        """Test that account reconciliation runs without errors."""
        assert isinstance(
            sample_account_reconciliation_mint_adjustment, ModelForecastingOutput
        )
        # Further checks can be added here to verify reconciliation results

        # check that gross_margin = net_revenue - cogs_total - t_w_total
        assert np.isclose(
            sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('gross_margin')
            ].arr,
            sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('net_revenue')
            ].arr
            - sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('cogs_total')
            ].arr
            - sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('t_w_total')
            ].arr,
        ).all()

        # check that total_costs = cogs_total + t_w_total
        assert np.isclose(
            sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('total_costs')
            ].arr,
            sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('cogs_total')
            ].arr
            + sample_account_reconciliation_mint_adjustment.accounts_forecasts[
                AccountType('t_w_total')
            ].arr,
        ).all()
        return
