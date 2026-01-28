from dataclasses import dataclass
from enum import Enum

from ...account_types import AccountType


class SpecialOperationsEnum(str, Enum):
    """
    Enumeration of special operations used in algebraic formulas.

    This enum defines special operators that have distinct semantic meaning
    in formula parsing and evaluation, separate from arithmetic operations.

    Attributes
    ----------
    EQUAL : str
        The equality operator ('=') used to separate left-hand side from
        right-hand side in formula expressions.

    Examples
    --------
    >>> SpecialOperationsEnum.EQUAL
    '='
    """

    EQUAL = '='


class SupportedOperationsEnum(str, Enum):
    """
    Enumeration of supported arithmetic operations for algebraic formulas.

    This enum defines the basic arithmetic operators that can be used in
    account reconciliation formulas. Each operation has an associated
    precedence level defined in OPERATION_TIERS.

    Attributes
    ----------
    ADDITION : str
        The addition operator ('+').
    SUBTRACTION : str
        The subtraction operator ('-').
    MULTIPLICATION : str
        The multiplication operator ('*').
    DIVISION : str
        The division operator ('/').

    See Also
    --------
    OPERATION_TIERS : Dictionary defining operator precedence levels.

    Examples
    --------
    >>> SupportedOperationsEnum.ADDITION
    '+'
    >>> SupportedOperationsEnum.MULTIPLICATION
    '*'
    """

    ADDITION = '+'
    SUBTRACTION = '-'
    MULTIPLICATION = '*'
    DIVISION = '/'


OPERATION_TIERS = {
    SupportedOperationsEnum.ADDITION: 1,
    SupportedOperationsEnum.SUBTRACTION: 1,
    SupportedOperationsEnum.MULTIPLICATION: 2,
    SupportedOperationsEnum.DIVISION: 2,
}
"""
Dictionary mapping arithmetic operations to their precedence levels.

This dictionary defines the operator precedence (order of operations) for
evaluating algebraic formulas. Higher tier numbers indicate higher precedence.
Operations in tier 2 (multiplication and division) are evaluated before
operations in tier 1 (addition and subtraction).

Keys
----
SupportedOperationsEnum
    The arithmetic operation.

Values
------
int
    The precedence level (1 = lower precedence, 2 = higher precedence).

Notes
-----
This follows standard mathematical operator precedence rules where
multiplication and division have higher precedence than addition and
subtraction.

Examples
--------
>>> OPERATION_TIERS[SupportedOperationsEnum.MULTIPLICATION]
2
>>> OPERATION_TIERS[SupportedOperationsEnum.ADDITION]
1
"""


@dataclass
class AlgebraicAccountFormula:
    """
    Represents an algebraic formula for account reconciliation.

    This dataclass encapsulates a formula expression that relates one account
    (left-hand side) to a combination of other accounts, operations, and
    constants (right-hand side). It is used in account reconciliation to
    define relationships between accounts.

    Attributes
    ----------
    lhs : AccountType
        The left-hand side of the formula, representing the target account
        whose value is defined by the formula.
    rhs : list[AccountType | SupportedOperationsEnum | float]
        The right-hand side of the formula, containing a sequence of account
        references, arithmetic operations, and numeric constants that define
        how to calculate the LHS account.

    Notes
    -----
    The RHS is stored as a list of tokens in infix notation. During evaluation,
    it is typically converted to postfix notation (Reverse Polish Notation) to
    properly handle operator precedence.

    Examples
    --------
    >>> formula = AlgebraicAccountFormula(
    ...     lhs=AccountType('TOTAL_REVENUE'),
    ...     rhs=[
    ...         AccountType('PRODUCT_A_REVENUE'),
    ...         SupportedOperationsEnum.ADDITION,
    ...         AccountType('PRODUCT_B_REVENUE')
    ...     ]
    ... )
    >>> # Represents: TOTAL_REVENUE = PRODUCT_A_REVENUE + PRODUCT_B_REVENUE
    """

    lhs: AccountType
    rhs: list[AccountType | SupportedOperationsEnum | float]
