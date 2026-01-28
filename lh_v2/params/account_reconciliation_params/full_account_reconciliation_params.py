from typing import Any

from pydantic import Field, model_validator

from lh_v2.datatypes.account_reconciliation_types import AccountReconciliationMethodEnum
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    AlgebraicAccountFormula,
)
from lh_v2.util import BaseParamsModel

from .algebraic_formula_parsing import parse_formula


class AccountReconciliationParams(BaseParamsModel):
    """
    Configuration parameters for account reconciliation process.

    Attributes
    ----------
    b_reconcile : bool
        Flag indicating whether to perform account reconciliation after forecasting.
        Default is True.
    method : str
        Method to use for reconciliation. Options include 'proportional_adjustment'
        and 'optimization_based'. Default is 'proportional_adjustment'.

    Examples
    --------
    >>> recon_params = AccountReconciliationParams()
    >>> recon_params.b_reconcile = False
    >>> recon_params.method = 'optimization_based'
    """

    b_reconcile: bool = False
    b_allow_default_algebraic_formulas: bool = True
    formulas: list[AlgebraicAccountFormula] = Field(default_factory=list)
    method: AccountReconciliationMethodEnum = (
        AccountReconciliationMethodEnum.ALGEBRAIC_FORMULA
    )

    @model_validator(mode='before')
    @classmethod
    def parse_formulas(cls, data: Any) -> Any:
        """
        Parse formula strings into lists of tokens before model validation.

        Converts:
            {'formulas': ['acc1 = acc2 + acc3']}
        To:
            {
                'formulas': [AlgebraicAccountFormula(
                    lhs=AccountType('acc1'),
                    rhs=[
                        AccountType('acc2'),
                        SupportedOperationsEnum.ADDITION,
                        AccountType('acc3')
                        ]
                    )
                ]
            }
        """
        if isinstance(data, dict) and 'formulas' in data:
            raw_formulas = data['formulas']

            # If formulas are strings, parse them
            if isinstance(raw_formulas, list) and raw_formulas:
                if isinstance(raw_formulas[0], str):
                    parsed_formulas: list[AlgebraicAccountFormula] = []
                    for formula_str in raw_formulas:
                        parsed = parse_formula(formula_str)
                        parsed_formulas.append(parsed)

                    data['formulas'] = parsed_formulas

        return data
