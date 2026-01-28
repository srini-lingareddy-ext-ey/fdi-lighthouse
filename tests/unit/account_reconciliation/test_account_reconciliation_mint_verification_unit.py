import pytest

from lh_v2.account_reconciliation.reconciliation_methods.mint_verification import (
    verify_mint_adjustment_possible,
)
from lh_v2.datatypes import AccountType
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    AlgebraicAccountFormula,
    SupportedOperationsEnum,
)


@pytest.mark.unit
@pytest.mark.account_reconciliation
class TestMintVerification:
    def test_verify_mint_adjustment_possible_all_valid(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result2'),
                rhs=[
                    AccountType('acc3'),
                    SupportedOperationsEnum.SUBTRACTION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc3'),
            AccountType('acc_result'),
            AccountType('acc_result2'),
        }
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is True

    def test_verify_mint_adjustment_possible_invalid_operation_account_multiplication(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.MULTIPLICATION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc_result'),
        }
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_invalid_operation_division_by_account(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.DIVISION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc_result'),
        }
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_missing_precalced(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc2')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_mixed_issues(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.MULTIPLICATION,
                    AccountType('acc2'),
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('total'),
                rhs=[
                    AccountType('acc3'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc3'),
            AccountType('total'),
        }
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_empty_formulas(self) -> None:
        formulas: list[AlgebraicAccountFormula] = []
        precalced_accounts = {AccountType('some_account')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is True

    def test_verify_mint_adjustment_possible_no_precalced_accounts(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts: set[AccountType] = set()
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_addition_of_float(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    100.0,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_subtraction_of_float(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.SUBTRACTION,
                    50.0,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_division_of_float(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.DIVISION,
                    2.0,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is True

    def test_verify_mint_adjustment_possible_multiplication_of_float(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.MULTIPLICATION,
                    3.0,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is True

    def test_verify_mint_adjustment_possible_only_float_in_formula(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    100.0,
                    SupportedOperationsEnum.DIVISION,
                    2.0,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_no_accounts_in_formula(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    100.0,
                    SupportedOperationsEnum.ADDITION,
                    50.0,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_single_account_no_operations(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is True

    def test_verify_mint_adjustment_possible_single_account_with_operations(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    SupportedOperationsEnum.SUBTRACTION,
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_multiple_accounts_no_operations(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc1'),
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc_result'),
        }
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_account_division_of_float(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    1.0,
                    SupportedOperationsEnum.DIVISION,
                    AccountType('acc1'),
                ],
            ),
        ]
        precalced_accounts = {AccountType('acc1'), AccountType('acc_result')}
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False

    def test_verify_mint_adjustment_possible_account_division_of_float_complex(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result'),
                rhs=[
                    AccountType('acc2'),
                    SupportedOperationsEnum.ADDITION,
                    2.0,
                    SupportedOperationsEnum.MULTIPLICATION,
                    1.0,
                    SupportedOperationsEnum.DIVISION,
                    AccountType('acc1'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc_result'),
        }
        assert verify_mint_adjustment_possible(formulas, precalced_accounts) is False
