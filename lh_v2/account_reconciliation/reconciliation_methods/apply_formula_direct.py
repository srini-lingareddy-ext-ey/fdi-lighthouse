import numpy as np

from lh_v2.datatypes import AccountGroupInfo, AccountType
from lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types import (
    OPERATION_TIERS,
    AlgebraicAccountFormula,
    IncorrectAccError,
    InvalidFormulaError,
    SupportedOperationsEnum,
)
from lh_v2.shared import ArrayF


def apply_algebraic_formulas_direct(
    forecast_data: AccountGroupInfo,
    formulas: list[AlgebraicAccountFormula],
) -> AccountGroupInfo:
    """
    Apply algebraic formulas directly to forecast data to compute new accounts.

    This function evaluates algebraic formulas to create new account forecasts
    based on existing accounts in the forecast data. The formulas are applied
    in order, and results are added to the account group.

    Parameters
    ----------
    forecast_data : AccountGroupInfo
        The existing forecast data containing account arrays and metadata.
    formulas : list[AlgebraicAccountFormula]
        List of algebraic formulas to apply. Each formula specifies how to
        compute a new account from existing accounts and constants.

    Returns
    -------
    AccountGroupInfo
        Updated forecast data with new accounts computed from formulas.
        Contains original accounts plus newly computed accounts.

    Notes
    -----
    The function creates a new array with additional rows for formula results
    and updates the account map to include the new accounts.
    """
    # Create array with space for existing accounts plus new formula results
    arr_new_accounts: ArrayF = np.zeros(
        (
            len(forecast_data) + len(formulas),
            forecast_data.arr.shape[1],
        ),
        dtype=forecast_data.arr.dtype,
    )

    # Copy existing account data to the new array
    arr_new_accounts[: len(forecast_data), :] = forecast_data.arr
    # Copy metadata for modification
    new_acc_map = forecast_data.account_map.copy()
    new_dates = list(forecast_data.dates).copy()

    # Start adding new accounts after existing ones
    acc_idx = len(forecast_data)
    for formula in formulas:
        # Compute the result array for this formula
        result_arr = apply_formula_arr(formula, forecast_data)
        arr_new_accounts[acc_idx, :] = result_arr

        # Register the new account in the account map using formula LHS as key
        new_acc_map[formula.lhs] = acc_idx
        # Add date entry for new account (assumes same date range as existing)
        new_dates.append(forecast_data.dates[0])
        acc_idx += 1

    # Return updated AccountGroupInfo with all accounts
    return AccountGroupInfo(
        arr=arr_new_accounts,
        account_map=new_acc_map,
        dates=new_dates,
        segment_type=forecast_data.segment_type,
        region_type=forecast_data.region_type,
        np_dtype=forecast_data.np_dtype,
    )


def apply_formula_arr(
    parsed_formula: AlgebraicAccountFormula,
    account_data: AccountGroupInfo,
) -> ArrayF:
    """
    Apply a single algebraic formula to account data arrays.

    Converts the formula to postfix notation and evaluates it to produce
    an array result representing the computed account values.

    Parameters
    ----------
    parsed_formula : AlgebraicAccountFormula
        The parsed formula containing LHS (result account) and RHS (expression).
    account_data : AccountGroupInfo
        Account data containing the arrays needed for formula evaluation.

    Returns
    -------
    ArrayF
        Computed array values for the formula result.

    See Also
    --------
    _convert_to_postfix : Converts infix expression to postfix notation.
    _evaluate_postfix : Evaluates postfix expression with account arrays.
    """
    # Convert formula RHS from infix to postfix for proper operator precedence
    postfix = _convert_to_postfix(parsed_formula.rhs)

    # Evaluate the postfix expression using account arrays
    result = _evaluate_postfix(postfix, account_data)

    return result


def apply_formula_float(
    parsed_rhs: list[SupportedOperationsEnum | float],
) -> float:
    """
    Apply an algebraic formula with only numeric operands.

    Evaluates a formula expression that contains only float values and
    operations, producing a single float result.

    Parameters
    ----------
    parsed_rhs : list[SupportedOperationsEnum | float]
        List of tokens representing the right-hand side of a formula,
        containing only operations and numeric constants.

    Returns
    -------
    float
        The computed numeric result of the formula.

    See Also
    --------
    _convert_to_postfix_float : Converts numeric expression to postfix.
    _evaluate_postfix_float : Evaluates postfix expression with floats.
    """
    # Convert numeric expression from infix to postfix notation
    postfix = _convert_to_postfix_float(parsed_rhs)

    # Evaluate the postfix expression with float arithmetic
    result = _evaluate_postfix_float(postfix)

    return result


def _convert_to_postfix(
    tokens: list[AccountType | SupportedOperationsEnum | float],
) -> list[AccountType | SupportedOperationsEnum | float]:
    """
    Convert infix notation to postfix (Reverse Polish Notation).

    Uses the Shunting Yard algorithm to convert an infix expression into
    postfix notation, respecting operator precedence.

    Parameters
    ----------
    tokens : list[AccountType | SupportedOperationsEnum | float]
        List of tokens in infix notation, including accounts, operations,
        and numeric constants.

    Returns
    -------
    list[AccountType | SupportedOperationsEnum | float]
        Tokens rearranged in postfix notation for proper evaluation order.

    Notes
    -----
    Higher precedence operators (multiplication, division) are evaluated
    before lower precedence operators (addition, subtraction).
    """
    output: list[AccountType | SupportedOperationsEnum | float] = []
    operator_stack: list[SupportedOperationsEnum] = []

    for token in tokens:
        if not isinstance(token, SupportedOperationsEnum):
            # Operands (accounts and numbers) go directly to output queue
            output.append(token)
        else:
            # Handle operators: pop higher/equal precedence operators first
            while operator_stack and OPERATION_TIERS.get(
                operator_stack[-1], 0
            ) >= OPERATION_TIERS.get(token, 0):
                output.append(operator_stack.pop())

            # Push current operator onto stack
            operator_stack.append(token)

    # Pop any remaining operators from stack to output
    while operator_stack:
        output.append(operator_stack.pop())

    return output


def _evaluate_postfix(
    postfix_tokens: list[AccountType | SupportedOperationsEnum | float],
    account_data: AccountGroupInfo,
) -> ArrayF:
    """
    Evaluate a postfix expression using account data arrays.

    Processes postfix tokens using a stack-based algorithm, performing
    operations on account arrays and numeric constants.

    Parameters
    ----------
    postfix_tokens : list[AccountType | SupportedOperationsEnum | float]
        Tokens in postfix notation to evaluate.
    account_data : AccountGroupInfo
        Account data providing arrays for account references.

    Returns
    -------
    ArrayF
        The resulting array from evaluating the postfix expression.

    Raises
    ------
    IncorrectAccError
        If a referenced account is not found in account_data.
    InvalidFormulaError
        If the expression is malformed (e.g., insufficient operands,
        invalid tokens, or improper stack state).

    Notes
    -----
    Operations are applied element-wise to account arrays. Division by
    zero is handled through standard NumPy behavior.
    """
    stack: list[ArrayF] = []

    for token in postfix_tokens:
        if not isinstance(token, SupportedOperationsEnum):
            # Process operands and push onto stack
            if type(token) is float:
                # Convert float constant to array filled with that value
                stack.append(
                    np.ones(account_data.arr.shape[1], dtype=account_data.np_dtype)
                    * token
                )
            elif type(token) is str:
                # Look up account by name and push its array
                if token not in account_data.get_ordered_accounts():
                    raise IncorrectAccError(
                        f"Account '{token}' not found in account_data"
                    )
                stack.append(account_data[token].arr.copy())
            else:
                raise InvalidFormulaError(f'Invalid token type: {type(token)}')

        else:
            # Process operators: pop two operands, apply operation, push result
            if len(stack) < 2:
                raise InvalidFormulaError(f"Not enough operands for operator '{token}'")

            # Pop in reverse order (right operand first, then left)
            right = stack.pop()
            left = stack.pop()

            # Apply the appropriate operation element-wise
            if token == SupportedOperationsEnum.ADDITION:
                result = left + right
            elif token == SupportedOperationsEnum.SUBTRACTION:
                result = left - right
            elif token == SupportedOperationsEnum.MULTIPLICATION:
                result = left * right
            elif token == SupportedOperationsEnum.DIVISION:
                result = left / right
                # Division by zero handled by NumPy (produces inf/nan)
                """with np.errstate(divide='ignore', invalid='ignore'):
                    result = np.divide(left, right)
                    result = np.where(np.isfinite(result), result, 0.0)"""
            else:
                raise InvalidFormulaError(f'Unknown operator: {token}')

            # Push result back onto stack
            stack.append(result)

    # Stack should contain exactly one element (the final result)
    if len(stack) != 1:
        raise InvalidFormulaError(
            f'Invalid expression evaluation. Stack has {len(stack)} items, expected 1'
        )

    return stack[0]


def _convert_to_postfix_float(
    tokens: list[SupportedOperationsEnum | float],
) -> list[SupportedOperationsEnum | float]:
    """
    Convert infix notation to postfix for float-only expressions.

    Similar to _convert_to_postfix but specialized for expressions
    containing only numeric values and operations.

    Parameters
    ----------
    tokens : list[SupportedOperationsEnum | float]
        List of tokens in infix notation, containing only operations
        and numeric constants.

    Returns
    -------
    list[SupportedOperationsEnum | float]
        Tokens rearranged in postfix notation.

    See Also
    --------
    _convert_to_postfix : General version supporting accounts.
    """
    output: list[SupportedOperationsEnum | float] = []
    operator_stack: list[SupportedOperationsEnum] = []

    for token in tokens:
        if not isinstance(token, SupportedOperationsEnum):
            # Numeric operands go directly to output queue
            output.append(token)
        else:
            # Pop operators with higher or equal precedence from stack
            while operator_stack and OPERATION_TIERS.get(
                operator_stack[-1], 0
            ) >= OPERATION_TIERS.get(token, 0):
                output.append(operator_stack.pop())

            # Push current operator onto stack
            operator_stack.append(token)

    # Pop remaining operators from stack to output
    while operator_stack:
        output.append(operator_stack.pop())

    return output


def _evaluate_postfix_float(
    postfix_tokens: list[SupportedOperationsEnum | float],
) -> float:
    """
    Evaluate a postfix expression containing only numeric values.

    Processes postfix tokens using a stack-based algorithm, performing
    operations on float values.

    Parameters
    ----------
    postfix_tokens : list[SupportedOperationsEnum | float]
        Tokens in postfix notation to evaluate, containing only
        operations and float values.

    Returns
    -------
    float
        The computed numeric result.

    Raises
    ------
    InvalidFormulaError
        If the expression is malformed (e.g., insufficient operands,
        unknown operators, or improper stack state).

    See Also
    --------
    _evaluate_postfix : General version supporting account arrays.
    """
    stack: list[float] = []

    for token in postfix_tokens:
        if not isinstance(token, SupportedOperationsEnum):
            # Push numeric operand onto stack
            stack.append(token)
        else:
            # Process operator: pop two operands, apply operation, push result
            if len(stack) < 2:
                raise InvalidFormulaError(f"Not enough operands for operator '{token}'")

            # Pop in reverse order (right operand first, then left)
            right = stack.pop()
            left = stack.pop()

            # Apply the appropriate arithmetic operation
            if token == SupportedOperationsEnum.ADDITION:
                result = left + right
            elif token == SupportedOperationsEnum.SUBTRACTION:
                result = left - right
            elif token == SupportedOperationsEnum.MULTIPLICATION:
                result = left * right
            elif token == SupportedOperationsEnum.DIVISION:
                result = left / right
            else:
                raise InvalidFormulaError(f'Unknown operator: {token}')

            # Push result back onto stack
            stack.append(result)

    # Stack should contain exactly one element (the final result)
    if len(stack) != 1:
        raise InvalidFormulaError(
            f'Invalid expression evaluation. Stack has {len(stack)} items, expected 1'
        )

    return stack[0]
