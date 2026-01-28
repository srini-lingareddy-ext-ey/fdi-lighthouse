from lh_v2.datatypes import AccountType
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    AlgebraicAccountFormula,
    SupportedOperationsEnum,
)


def verify_mint_adjustment_possible(
    formulas: list[AlgebraicAccountFormula],
    precalced_accounts: set[AccountType],
) -> bool:
    """
    Verify that MINT adjustment reconciliation is possible given the formulas
    and previously calculated accounts.

    Parameters
    ----------
    formulas : list[AlgebraicAccountFormula]
        List of algebraic account formulas to verify.
    precalced_accounts : set[AccountType]
        Set of account types that have been previously calculated.

    Returns
    -------
    bool
        True if all formulas can be reconciled with the precalculated accounts,
        False otherwise.

    Notes
    -----
    MINT adjustment requires that formulas only contain account additions
    and that all referenced accounts have been precalculated.
    """
    # Check that each formula only contains addition/subtraction of accounts
    for formula in formulas:
        if not _verify_only_acc_addition(formula):
            return False
    # Verify all accounts referenced in formulas are available
    return _verify_accounts_precalced(formulas, precalced_accounts)


def _verify_accounts_precalced(
    formulas: list[AlgebraicAccountFormula],
    precalced_accounts: set[AccountType],
) -> bool:
    """
    Verify that all accounts referenced in formulas have been precalculated.

    This function checks that both left-hand side (LHS) accounts and right-hand
    side (RHS) accounts are either precalculated or appear as LHS in another formula.

    Parameters
    ----------
    formulas : list[AlgebraicAccountFormula]
        List of algebraic account formulas to verify.
    precalced_accounts : set[AccountType]
        Set of account types that have been previously calculated.

    Returns
    -------
    bool
        True if all referenced accounts are available, False otherwise.
    """
    # Collect all LHS accounts from formulas
    lhs_accounts: set[AccountType] = set()
    for formula in formulas:
        lhs_accounts.add(formula.lhs)
        # Each LHS account must exist in precalculated accounts
        if formula.lhs not in precalced_accounts:
            return False

    # Extract all accounts from RHS of all formulas
    rhs_accounts = _get_accounts_in_rhs(formulas)
    # Each RHS account must either be precalculated or appear as LHS in another formula
    for acc in rhs_accounts:
        if acc not in precalced_accounts and acc not in lhs_accounts:
            return False
    return True


def _get_accounts_in_rhs(
    formulas: list[AlgebraicAccountFormula],
) -> set[AccountType]:
    """
    Extract all account types from the right-hand side of formulas.

    Parameters
    ----------
    formulas : list[AlgebraicAccountFormula]
        List of algebraic account formulas to extract accounts from.

    Returns
    -------
    set[AccountType]
        Set of all unique account types found in the RHS of all formulas.
        Excludes operations and numeric constants.
    """
    accounts_in_rhs: set[AccountType] = set()
    for formula in formulas:
        # Iterate through all tokens in the RHS
        for token in formula.rhs:
            # Filter out operations and numeric constants, keep only accounts
            if (
                not isinstance(token, SupportedOperationsEnum)
                and not isinstance(token, float)
                and not isinstance(token, int)
            ):
                accounts_in_rhs.add(token)
    return accounts_in_rhs


def _verify_only_acc_addition(formula: AlgebraicAccountFormula) -> bool:
    """
    Verify that a formula only contains account additions/subtractions.

    This function ensures that the formula follows MINT adjustment rules:
    - Accounts can only be combined with addition or subtraction
    - Multiplication/division can only be applied to entire accounts, not between them
    - At least one account must exist in the RHS

    Parameters
    ----------
    formula : AlgebraicAccountFormula
        The formula to verify.

    Returns
    -------
    bool
        True if the formula only uses account addition/subtraction, False otherwise.
    """
    # Track operations and constants between accounts
    between_accs: list[SupportedOperationsEnum | float] = []
    n_accs: int = 0

    # Process each token in the RHS
    for token in formula.rhs:
        if isinstance(token, SupportedOperationsEnum):
            # Accumulate operations between accounts
            between_accs.append(token)
        elif isinstance(token, float):
            # Accumulate numeric constants between accounts
            between_accs.append(token)
        else:
            # Token is an account - verify the operations before it
            if not _verify_between_accs(between_accs, n_accs):
                return False
            # Reset for next account
            between_accs = []
            n_accs += 1

    # Ensure at least one account exists in the RHS
    if n_accs == 0:
        return False

    # Verify operations after the last account
    return _verify_between_accs(between_accs, -1)


def _verify_between_accs(
    between_accs: list[SupportedOperationsEnum | float],
    n_accs: int,
) -> bool:
    """
    Verify operations and constants between accounts follow MINT rules.

    This function validates the tokens that appear between accounts in a formula,
    ensuring they conform to MINT adjustment requirements.

    Parameters
    ----------
    between_accs : list[SupportedOperationsEnum | float]
        List of operations and numeric constants between accounts.
    n_accs : int
        Position indicator: 0 for before first account, -1 for after last account,
        or positive integer for between accounts.

    Returns
    -------
    bool
        True if the operations between accounts are valid for MINT adjustment,
        False otherwise.

    Notes
    -----
    - Before the first account (n_accs=0): No addition/subtraction allowed,
      no division by an account
    - After the last account (n_accs=-1): No addition/subtraction allowed
    - Between accounts (n_accs>0): Exactly one addition/subtraction required,
      no division by an account
    """
    # This assumes there is an account in the full formula
    if n_accs == 0:
        # Before first account
        if len(between_accs) == 0:
            # No operations before first account is valid
            return True

        # Addition/subtraction before first account is not allowed (e.g., "5 + Account1")
        if (SupportedOperationsEnum.ADDITION in between_accs) or (
            SupportedOperationsEnum.SUBTRACTION in between_accs
        ):
            return False

        # Division must not have an account as divisor (last operation before account)
        if between_accs[-1] == SupportedOperationsEnum.DIVISION:
            return False

    elif n_accs == -1:
        # After last account
        if len(between_accs) == 0:
            # No operations after last account is valid
            return True

        # Addition/subtraction after last account is not allowed (e.g., "Account1 + 5")
        if (SupportedOperationsEnum.ADDITION in between_accs) or (
            SupportedOperationsEnum.SUBTRACTION in between_accs
        ):
            return False
    else:
        # Between accounts (n_accs > 0)
        if len(between_accs) == 0:
            # Operations must exist between accounts
            return False

        # Count addition/subtraction operations between accounts
        n_add_sub = 0
        for token in between_accs:
            if (
                token == SupportedOperationsEnum.ADDITION
                or token == SupportedOperationsEnum.SUBTRACTION
            ):
                n_add_sub += 1
        # Exactly one addition/subtraction required between accounts
        if n_add_sub == 0 or n_add_sub > 1:
            return False

        # Division must not have an account as divisor (last operation before next account)
        if between_accs[-1] == SupportedOperationsEnum.DIVISION:
            return False
    return True
