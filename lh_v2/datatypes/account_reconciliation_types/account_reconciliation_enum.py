from enum import Enum


class AccountReconciliationMethodEnum(str, Enum):
    """Enumeration of account reconciliation methods."""

    ALGEBRAIC_FORMULA = 'algebraic_formula'
    MINT_ADJUSTMENT = 'mint_adjustment'
