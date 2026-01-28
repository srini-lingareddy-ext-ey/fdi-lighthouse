import pytest

from lh_v2.datatypes import AccountType
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    IncompleteFormulaError,
    IncorrectAccError,
    InvalidLHSFormulaError,
    InvalidOrderingError,
    MultipleEqualsError,
    SupportedOperationsEnum,
)
from lh_v2.params.account_reconciliation_params.algebraic_formula_parsing import (
    parse_formula,
    validate_formulas_ordered,
    verify_formula_accounts,
)


@pytest.mark.unit
@pytest.mark.account_reconciliation_params
class TestInitialFormulaParsing:
    def test_valid_formula(self) -> None:
        formula = 'acc_result = acc1 + acc2 - 100.5'
        parsed = parse_formula(formula)
        assert parsed.lhs == 'acc_result'
        assert parsed.rhs == [
            'acc1',
            SupportedOperationsEnum.ADDITION,
            'acc2',
            SupportedOperationsEnum.SUBTRACTION,
            100.5,
        ]

    def test_valid_formula2(self) -> None:
        formula = 'total = acc1 + acc2 + 1.0 / acc3 - 50'
        parsed = parse_formula(formula)
        assert parsed.lhs == 'total'
        assert parsed.rhs == [
            'acc1',
            SupportedOperationsEnum.ADDITION,
            'acc2',
            SupportedOperationsEnum.ADDITION,
            1.0,
            SupportedOperationsEnum.DIVISION,
            'acc3',
            SupportedOperationsEnum.SUBTRACTION,
            50.0,
        ]

    def test_incomplete_formula_error(self) -> None:
        formula = 'acc_result ='
        with pytest.raises(IncompleteFormulaError):
            parse_formula(formula)

    def test_invalid_lhs_formula_error(self) -> None:
        formula = 'acc_result extra = acc1 + acc2'
        with pytest.raises(InvalidLHSFormulaError):
            parse_formula(formula)

    def test_invalid_lhs_formula_error_operator_first(self) -> None:
        formula = '+ acc1 + acc2 = acc_result'
        with pytest.raises(InvalidLHSFormulaError):
            parse_formula(formula)

    def test_invalid_lhs_formula_error_equals_first(self) -> None:
        formula = '= acc1 + acc2 + acc3'
        with pytest.raises(InvalidLHSFormulaError):
            parse_formula(formula)

    def test_invalid_lhs_formula_error_number_first(self) -> None:
        formula = '100 + acc1 = acc_result'
        with pytest.raises(InvalidLHSFormulaError):
            parse_formula(formula)

    def test_multiple_equals_error(self) -> None:
        formula = 'acc_result = acc1 + acc2 = acc3'
        with pytest.raises(MultipleEqualsError):
            parse_formula(formula)

    def test_invalid_ordering_error_operator_after_equals(self) -> None:
        formula = 'acc3 = + acc1 - acc2'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)

    def test_invalid_ordering_error_number_account(self) -> None:
        formula = 'acc3 = acc1 100 + acc2'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)

    def test_invalid_ordering_error_consecutive_operators(self) -> None:
        formula = 'acc3 = acc1 + - acc2'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)

    def test_invalid_ordering_error_consecutive_numbers(self) -> None:
        formula = 'acc3 = acc1 + 100 50 - acc2'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)

    def test_invalid_ordering_error_ends_with_operator(self) -> None:
        formula = 'acc3 = acc1 + acc2 -'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)

    def test_invalid_ordering_error_rhs_starts_with_operator(self) -> None:
        formula = 'acc3 = + acc1 + acc2'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)

    def test_invalid_ordering_error_consecutive_accounts(self) -> None:
        formula = 'acc3 = acc1 acc2 + 100'
        with pytest.raises(InvalidOrderingError):
            parse_formula(formula)


@pytest.mark.unit
@pytest.mark.account_reconciliation_params
class TestVerifyFormulaAccounts:
    def test_verify_formula_accounts_valid(self) -> None:
        formula = parse_formula('acc_result = acc1 + acc2 - 50')
        available_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc3'),
        }
        verify_formula_accounts(formula, available_accounts)  # Should not raise

    def test_verify_formula_accounts_invalid(self) -> None:
        formula = parse_formula('acc_result = acc1 + acc4 - 50')
        available_accounts = {
            AccountType('acc1'),
            AccountType('acc2'),
            AccountType('acc3'),
        }
        with pytest.raises(IncorrectAccError):
            verify_formula_accounts(formula, available_accounts)


@pytest.mark.unit
@pytest.mark.account_reconciliation_params
class TestValidateFormulasOrdered:
    def test_validate_formulas_ordered_valid(self) -> None:
        formulas = [
            parse_formula('acc1 = 100'),
            parse_formula('acc2 = acc1 + 50'),
            parse_formula('acc3 = acc2 - 25'),
        ]
        available_accounts = set()
        validate_formulas_ordered(available_accounts, formulas)  # Should not raise

    def test_validate_formulas_ordered_invalid(self) -> None:
        formulas = [
            parse_formula('acc2 = acc1 + 50'),
            parse_formula('acc1 = 100'),
        ]
        available_accounts = set()
        with pytest.raises(IncorrectAccError):
            validate_formulas_ordered(
                available_accounts, formulas
            )  # acc1 not available when parsing first formula

    def test_validate_formulas_ordered_invalid_account(self) -> None:
        formulas = [
            parse_formula('acc1 = 100'),
            parse_formula('acc2 = acc1 + acc4'),
        ]
        available_accounts = set()
        with pytest.raises(IncorrectAccError):
            validate_formulas_ordered(
                available_accounts, formulas
            )  # acc4 not available when parsing second formula

    def test_validate_formulas_ordered_empty(self) -> None:
        formulas = []
        available_accounts = set()
        validate_formulas_ordered(available_accounts, formulas)  # Should not raise

    def test_validate_formulas_ordered_single(self) -> None:
        formulas = [parse_formula('acc1 = 100')]
        available_accounts = set()
        validate_formulas_ordered(available_accounts, formulas)  # Should not raise

    def test_validate_formulas_ordered_available_accounts(self) -> None:
        formulas = [parse_formula('acc2 = acc1 + 50')]
        available_accounts = {AccountType('acc1')}
        validate_formulas_ordered(available_accounts, formulas)  # Should not raise

    def test_validate_formulas_ordered_reuse_account(self) -> None:
        formulas = [
            parse_formula('acc1 = 100'),
            parse_formula('acc2 = acc1 + 50'),
            parse_formula('acc3 = acc1 + acc2 - 25'),
        ]
        available_accounts = set()
        validate_formulas_ordered(available_accounts, formulas)  # Should not raise
