import numpy as np
import pytest

from lh_v2.account_reconciliation.reconciliation_methods.apply_formula_mint import (
    parse_formula_mint,
    parse_formulas_mint,
)
from lh_v2.datatypes import AccountType
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    AlgebraicAccountFormula,
    SupportedOperationsEnum,
)
from lh_v2.shared import BASE_NP_DTYPE


@pytest.mark.unit
@pytest.mark.account_reconciliation
class TestMintFormulaParsing:
    def test_parse_formula_mint_valid(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result'): 3,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.ADDITION,
                AccountType('acc2'),
                SupportedOperationsEnum.SUBTRACTION,
                AccountType('acc3'),
            ],
        )
        base_accounts = {AccountType('acc1'), AccountType('acc2'), AccountType('acc3')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(
            parsed_u, np.array([[-1, -1, 1, 1]], dtype=BASE_NP_DTYPE)
        ).all()
        assert np.isclose(parsed_s, np.array([[1, 1, -1]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_with_constants(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result'): 3,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.ADDITION,
                100.0,
                SupportedOperationsEnum.MULTIPLICATION,
                AccountType('acc2'),
                SupportedOperationsEnum.DIVISION,
                2.0,
                SupportedOperationsEnum.ADDITION,
                AccountType('acc3'),
            ],
        )
        base_accounts = {AccountType('acc1'), AccountType('acc2'), AccountType('acc3')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(
            parsed_u, np.array([[-1, -50.0, -1, 1]], dtype=BASE_NP_DTYPE)
        ).all()
        assert np.isclose(parsed_s, np.array([[1, 50.0, 1]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_unused_base_account(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result'): 3,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.ADDITION,
                AccountType('acc2'),
            ],
        )
        base_accounts = {AccountType('acc1'), AccountType('acc2'), AccountType('acc3')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(
            parsed_u, np.array([[-1, -1, 0, 1]], dtype=BASE_NP_DTYPE)
        ).all()
        assert np.isclose(parsed_s, np.array([[1, 1, 0]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_unused_final_account(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result'): 3,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.ADDITION,
                AccountType('acc2'),
            ],
        )
        base_accounts = {AccountType('acc1'), AccountType('acc2')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(
            parsed_u, np.array([[-1, -1, 0, 1]], dtype=BASE_NP_DTYPE)
        ).all()
        assert np.isclose(parsed_s, np.array([[1, 1]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_single_account(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc_result'): 1,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[AccountType('acc1')],
        )
        base_accounts = {AccountType('acc1')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(parsed_u, np.array([[-1, 1]], dtype=BASE_NP_DTYPE)).all()
        assert np.isclose(parsed_s, np.array([[1]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_complex_multiplication_division(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc_result'): 2,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.MULTIPLICATION,
                3.0,
                SupportedOperationsEnum.DIVISION,
                2.0,
                SupportedOperationsEnum.ADDITION,
                AccountType('acc2'),
            ],
        )
        base_accounts = {AccountType('acc1'), AccountType('acc2')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(
            parsed_u, np.array([[-1.5, -1, 1]], dtype=BASE_NP_DTYPE)
        ).all()
        assert np.isclose(parsed_s, np.array([[1.5, 1]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_all_subtraction(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result'): 3,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.SUBTRACTION,
                AccountType('acc2'),
                SupportedOperationsEnum.SUBTRACTION,
                AccountType('acc3'),
            ],
        )
        base_accounts = {AccountType('acc1'), AccountType('acc2'), AccountType('acc3')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(
            parsed_u, np.array([[-1, 1, 1, 1]], dtype=BASE_NP_DTYPE)
        ).all()
        assert np.isclose(parsed_s, np.array([[1, -1, -1]], dtype=BASE_NP_DTYPE)).all()

    def test_parse_formula_mint_only_constants(self) -> None:
        account_inds = {
            AccountType('acc1'): 0,
            AccountType('acc_result'): 1,
        }
        formula = AlgebraicAccountFormula(
            lhs=AccountType('acc_result'),
            rhs=[
                AccountType('acc1'),
                SupportedOperationsEnum.MULTIPLICATION,
                0.0,
            ],
        )
        base_accounts = {AccountType('acc1')}
        parsed_u, parsed_s = parse_formula_mint(
            account_ind=account_inds, base_accounts=base_accounts, formula=formula
        )

        assert np.isclose(parsed_u, np.array([[0, 1]], dtype=BASE_NP_DTYPE)).all()
        assert np.isclose(parsed_s, np.array([[0]], dtype=BASE_NP_DTYPE)).all()


@pytest.mark.unit
@pytest.mark.account_reconciliation
class TestMintFormulasParsing:
    def test_parse_formulas_mint_valid_shape(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result2'),
                rhs=[
                    AccountType('acc2'),
                    SupportedOperationsEnum.SUBTRACTION,
                    AccountType('acc3'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result1'): 3,
            AccountType('acc_result2'): 4,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        assert parsed_S.shape == (5, 3)
        assert parsed_J.shape == (3, 5)
        assert parsed_U.shape == (5, 2)

    def test_parse_formulas_mint_single_formula(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc_result1'): 2,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        assert parsed_S.shape == (3, 2)
        assert parsed_J.shape == (2, 3)
        assert parsed_U.shape == (3, 1)

    def test_parse_formulas_mint_valid_matricies(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result2'),
                rhs=[
                    AccountType('acc2'),
                    SupportedOperationsEnum.SUBTRACTION,
                    AccountType('acc3'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result1'): 3,
            AccountType('acc_result2'): 4,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        expected_S = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [0, 1, -1],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_J = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_U = np.array(
            [
                [-1, -1, 0, 1, 0],
                [0, -1, 1, 0, 1],
            ],
            dtype=BASE_NP_DTYPE,
        ).T

        assert np.isclose(parsed_S, expected_S).all()
        assert np.isclose(parsed_J, expected_J).all()
        assert np.isclose(parsed_U, expected_U).all()

    def test_parse_formulas_mint_floats_in_formula(
        self,
    ) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    100.0,
                    SupportedOperationsEnum.MULTIPLICATION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc_result1'): 2,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        expected_S = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 100.0],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_J = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_U = np.array(
            [
                [-1, -100.0, 1],
            ],
            dtype=BASE_NP_DTYPE,
        ).T

        assert np.isclose(parsed_S, expected_S).all()
        assert np.isclose(parsed_J, expected_J).all()
        assert np.isclose(parsed_U, expected_U).all()

    def test_parse_formulas_mint_float_division(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                    SupportedOperationsEnum.DIVISION,
                    5.0,
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc_result1'): 2,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        expected_S = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 0.2],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_J = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_U = np.array(
            [
                [-1, -0.2, 1],
            ],
            dtype=BASE_NP_DTYPE,
        ).T

        assert np.isclose(parsed_S, expected_S).all()
        assert np.isclose(parsed_J, expected_J).all()
        assert np.isclose(parsed_U, expected_U).all()

    def test_parse_formulas_mint_multiple_formulas_same_base_accounts(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.MULTIPLICATION,
                    2.0,
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result2'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.DIVISION,
                    2.0,
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc_result1'): 1,
            AccountType('acc_result2'): 2,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        expected_S = np.array(
            [
                [1],
                [2.0],
                [0.5],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_J = np.array(
            [
                [1, 0, 0],
            ],
            dtype=BASE_NP_DTYPE,
        )
        expected_U = np.array(
            [
                [-2.0, 1, 0],
                [-0.5, 0, 1],
            ],
            dtype=BASE_NP_DTYPE,
        ).T

        assert np.isclose(parsed_S, expected_S).all()
        assert np.isclose(parsed_J, expected_J).all()
        assert np.isclose(parsed_U, expected_U).all()

    def test_parse_formulas_mint_complex_multi_formula(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                    SupportedOperationsEnum.MULTIPLICATION,
                    2.0,
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result2'),
                rhs=[
                    AccountType('acc3'),
                    SupportedOperationsEnum.SUBTRACTION,
                    AccountType('acc1'),
                    SupportedOperationsEnum.DIVISION,
                    4.0,
                ],
            ),
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result3'),
                rhs=[
                    AccountType('acc2'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc3'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc_result1'): 3,
            AccountType('acc_result2'): 4,
            AccountType('acc_result3'): 5,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        assert parsed_S.shape == (6, 3)
        assert parsed_J.shape == (3, 6)
        assert parsed_U.shape == (6, 3)

    def test_parse_formulas_mint_unused_accounts_in_index(self) -> None:
        formulas = [
            AlgebraicAccountFormula(
                lhs=AccountType('acc_result1'),
                rhs=[
                    AccountType('acc1'),
                    SupportedOperationsEnum.ADDITION,
                    AccountType('acc2'),
                ],
            ),
        ]
        precalced_accounts = {
            AccountType('acc1'): 0,
            AccountType('acc2'): 1,
            AccountType('acc3'): 2,
            AccountType('acc4'): 3,
            AccountType('acc_result1'): 4,
        }
        parsed_S, parsed_J, parsed_U = parse_formulas_mint(
            account_ind=precalced_accounts, formulas=formulas
        )

        assert parsed_S.shape == (5, 4)
        assert parsed_J.shape == (4, 5)
        assert parsed_U.shape == (5, 1)
