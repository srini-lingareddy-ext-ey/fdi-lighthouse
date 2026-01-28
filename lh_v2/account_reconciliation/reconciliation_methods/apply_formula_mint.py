import numpy as np

from lh_v2.datatypes import AccountGroupInfo, AccountType
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    AlgebraicAccountFormula,
    InvalidFormulaError,
    SupportedOperationsEnum,
)
from lh_v2.shared import BASE_NP_DTYPE, ArrayF

from .apply_formula_direct import apply_formula_float


def apply_formulas_mint(
    accounts_forecasts: AccountGroupInfo,
    formulas: list[AlgebraicAccountFormula],
    validation_errors: dict[AccountType, ArrayF],
) -> AccountGroupInfo:
    """
    Apply algebraic formulas using MINT (Minimum Trace) adjustment method.

    This function reconciles account forecasts using the MINT approach, which
    minimizes the trace of the forecast error covariance matrix while ensuring
    accounting equations are satisfied.

    Parameters
    ----------
    accounts_forecasts : AccountGroupInfo
        The existing account forecasts to be reconciled.
    formulas : list[AlgebraicAccountFormula]
        List of algebraic formulas representing accounting constraints.
    validation_errors : dict[AccountType, ArrayF]
        Historical validation errors for each account, used to estimate
        the error covariance matrix.

    Returns
    -------
    AccountGroupInfo
        Reconciled account forecasts that satisfy the specified formulas
        while minimizing forecast error variance.

    Notes
    -----
    The MINT method computes an optimal projection matrix P_opt that
    reconciles forecasts while accounting for forecast error correlations.
    If no formulas are provided, returns the original forecasts unchanged.

    References
    ----------
    Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
    Optimal forecast reconciliation for hierarchical and grouped time series
    through trace minimization. Journal of the American Statistical Association.
    """
    # If no formulas provided, no reconciliation needed
    if len(formulas) == 0:
        return accounts_forecasts

    # Parse formulas into matrices: S (summing), J (selection), U (constraints)
    S, J, U = parse_formulas_mint(
        accounts_forecasts.account_map,
        formulas,
    )

    # Compute error covariance matrix W from historical validation errors
    W = np.cov(
        np.vstack(
            [
                validation_errors[acc]
                for acc in accounts_forecasts.get_ordered_accounts()
            ]
        ),
        dtype=accounts_forecasts.np_dtype,
    )

    # Compute optimal projection matrix using MINT formula
    # P_opt minimizes trace of reconciled forecast error covariance
    P_opt = J - J @ W @ U @ np.linalg.inv(U.T @ W @ U) @ U.T

    # Apply reconciliation: reconciled = S @ P_opt @ original forecasts
    new_arr = (S @ P_opt @ accounts_forecasts.arr).astype(accounts_forecasts.np_dtype)

    # Return reconciled forecasts with updated array
    return AccountGroupInfo(
        arr=new_arr,
        account_map=accounts_forecasts.account_map,
        dates=accounts_forecasts.dates,
        segment_type=accounts_forecasts.segment_type,
        region_type=accounts_forecasts.region_type,
        np_dtype=accounts_forecasts.np_dtype,
    )


def parse_formulas_mint(
    account_ind: dict[AccountType, int],
    formulas: list[AlgebraicAccountFormula],
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """
    Parse formulas into matrices for MINT reconciliation.

    Constructs the summing matrix S, selection matrix J, and constraint
    matrix U needed for MINT reconciliation from algebraic formulas.

    Parameters
    ----------
    account_ind : dict[AccountType, int]
        Mapping from account types to their indices in the data array.
    formulas : list[AlgebraicAccountFormula]
        List of algebraic formulas to parse.

    Returns
    -------
    S : ArrayF
        Summing matrix that defines how base accounts combine to form
        aggregate accounts (shape: n_total x n_base).
    J : ArrayF
        Selection matrix that extracts base-level forecasts
        (shape: n_base x n_total).
    U : ArrayF
        Constraint matrix representing the linear constraints imposed
        by the formulas (shape: n_total x n_formulas).

    Notes
    -----
    Base accounts are those that do not appear as LHS in any formula.
    The matrices S, J, and U together define the reconciliation problem:
    - S maps base forecasts to all forecasts
    - J selects only base forecasts
    - U defines the constraint structure
    """
    # Identify base accounts (those not derived from formulas)
    base_accounts: set[AccountType] = set(account_ind.keys())
    for formula in formulas:
        # Remove LHS accounts from base set (they're aggregate/derived accounts)
        base_accounts.discard(formula.lhs)

    # Parse each formula into matrix rows
    S_rows: list[ArrayF] = []
    U_rows: list[ArrayF] = []
    for formula in formulas:
        # Extract constraint and summing rows for this formula
        u_row, s_row = parse_formula_mint(
            account_ind,
            base_accounts,
            formula,
        )
        U_rows.append(u_row)
        S_rows.append(s_row)

    # Build summing matrix S: [I_base; S_aggregate]
    # Identity block for base accounts, formula rows for aggregates
    S = np.vstack([np.eye(len(base_accounts)), *S_rows]).astype(BASE_NP_DTYPE)

    # Build selection matrix J: [I_base | 0]
    # Selects only base-level forecasts from reconciled output
    J = np.hstack(
        [np.eye(len(base_accounts)), np.zeros((len(base_accounts), len(formulas)))]
    )

    # Build constraint matrix U by stacking rows and transposing
    U = np.vstack(U_rows).astype(BASE_NP_DTYPE).T
    return S, J, U


def parse_formula_mint(
    account_ind: dict[AccountType, int],
    base_accounts: set[AccountType],
    formula: AlgebraicAccountFormula,
) -> tuple[ArrayF, ArrayF]:
    """
    Parse a single formula into matrix rows for MINT reconciliation.

    Extracts coefficients from a formula and constructs corresponding rows
    for the constraint matrix U and summing matrix S.

    Parameters
    ----------
    account_ind : dict[AccountType, int]
        Mapping from account types to their indices in the data array.
    base_accounts : set[AccountType]
        Set of base-level accounts (those not derived from formulas).
    formula : AlgebraicAccountFormula
        The algebraic formula to parse.

    Returns
    -------
    u_row : ArrayF
        Row vector for the constraint matrix U (shape: 1 x n_total).
        Represents the linear constraint imposed by this formula.
    s_row : ArrayF
        Row vector for the summing matrix S (shape: 1 x n_base).
        Defines how base accounts combine to form this aggregate account.

    Notes
    -----
    The function assumes the formula satisfies MINT constraints:
    - Only addition/subtraction between accounts
    - No accounts as divisors
    - All referenced accounts exist in account_ind

    The formula is decomposed into sub-formulas separated by addition/
    subtraction, with coefficients extracted for each account.
    """
    # what are the guarantees about the formula once it is here?
    # - there is only one addition/subtraction between accounts
    # - lhs is in account_ind
    # - all accounts in rhs are in account_ind
    # - no accounts are acting as a divisor
    # - no floating floats (soemthing that evaluates to float surrounded by parentheses)
    #       (as in acc1 + 2.5 + 2 * acc2) the 2.5 is floating
    # idea:
    # break rhs into segments between add/subtract
    #   each segment should have a single account in it
    # congregate all the coefficients for each account
    #   including converting subtraction to add mult -1
    # build the row of the matrix accordingly

    # Split formula RHS into sub-formulas at addition/subtraction points
    # Each sub-formula should contain exactly one account with its coefficient
    sub_formulas: list[list[SupportedOperationsEnum | AccountType | float]] = []
    current_subformula: list[SupportedOperationsEnum | AccountType | float] = []
    for token in formula.rhs:
        if token == SupportedOperationsEnum.ADDITION:
            # Save current sub-formula and start new one
            sub_formulas.append(current_subformula)
            current_subformula = []
        elif token == SupportedOperationsEnum.SUBTRACTION:
            # Save current sub-formula and start new one with negation
            sub_formulas.append(current_subformula)
            # Convert subtraction to addition of negative (multiply by -1)
            current_subformula = [-1.0, SupportedOperationsEnum.MULTIPLICATION]
        else:
            # Accumulate tokens in current sub-formula
            current_subformula.append(token)
    # Add the final sub-formula
    sub_formulas.append(current_subformula)

    # Extract coefficient for each account from sub-formulas
    local_account_coeffs: dict[AccountType, float] = {}
    for sub_formula in sub_formulas:
        # Replace account with 1.0 and evaluate to get coefficient
        account, new_sub_formula = _replace_acc_in_sub_formula(sub_formula)
        # Evaluate numeric expression to compute the coefficient
        local_account_coeffs[account] = apply_formula_float(new_sub_formula)

    # Build s_row: coefficients for base accounts in summing matrix
    s_row = np.zeros((1, len(base_accounts)), dtype=BASE_NP_DTYPE)
    for acc in base_accounts:
        # Set coefficient for each base account (0 if not in formula)
        s_row[0, account_ind[acc]] = local_account_coeffs.get(acc, 0.0)

    # Build u_row: constraint vector (LHS - RHS = 0)
    u_row = np.zeros((1, len(account_ind)), dtype=BASE_NP_DTYPE)
    for acc in account_ind.keys():
        if acc == formula.lhs:
            # LHS account gets coefficient +1
            u_row[0, account_ind[acc]] = 1.0
        else:
            # RHS accounts get negative of their coefficients
            u_row[0, account_ind[acc]] = -1.0 * local_account_coeffs.get(acc, 0.0)

    return u_row, s_row


def _replace_acc_in_sub_formula(
    sub_formula: list[SupportedOperationsEnum | AccountType | float],
) -> tuple[AccountType, list[SupportedOperationsEnum | float]]:
    """
    Replace account reference in sub-formula with coefficient 1.0.

    Extracts the single account from a sub-formula and replaces it with
    a coefficient of 1.0, allowing the sub-formula to be evaluated as
    a numeric expression.

    Parameters
    ----------
    sub_formula : list[SupportedOperationsEnum | AccountType | float]
        Sub-formula tokens containing exactly one account reference along
        with operations and numeric constants.

    Returns
    -------
    account : AccountType
        The account type found in the sub-formula.
    new_sub_formula : list[SupportedOperationsEnum | float]
        The sub-formula with the account replaced by 1.0, ready for
        numeric evaluation to extract the coefficient.

    Raises
    ------
    InvalidFormulaError
        If the sub-formula does not contain exactly one account reference.

    Examples
    --------
    >>> _replace_acc_in_sub_formula([2.0, '*', 'REVENUE'])
    ('REVENUE', [2.0, '*', 1.0])
    """
    account: AccountType | None = None
    new_sub_formula: list[SupportedOperationsEnum | float] = []

    # Count accounts to ensure exactly one exists
    c_accounts: int = 0
    for token in sub_formula:
        # Check if token is an account (string that's not an operation)
        if not isinstance(token, SupportedOperationsEnum) and isinstance(token, str):
            # Found an account - save it and replace with 1.0
            account = token
            new_sub_formula.append(1.0)  # Replace account with coefficient 1.0
            c_accounts += 1
        else:
            # Keep operations and numeric constants as-is
            new_sub_formula.append(token)

    # Validate exactly one account was found
    if account is None or c_accounts != 1:
        raise InvalidFormulaError(f'Invalid sub-formula: {sub_formula}')
    return account, new_sub_formula
