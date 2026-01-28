import builtins

import lh_v2.datatypes.account_reconciliation_types.algebraic_formula_types as alg_formlula_types
from lh_v2.datatypes import AccountType


def parse_formula(
    formula: str,
) -> alg_formlula_types.AlgebraicAccountFormula:
    # Split formula by spaces and filter out empty tokens
    formula_split = formula.split(' ')
    tokens = [token for token in formula_split if token]

    # Validate length greater than 2
    if len(tokens) < 3:
        raise alg_formlula_types.IncompleteFormulaError(
            f"'{formula}'. Expected format: 'acc_result = acc1 + acc2 - acc3'"
        )

    # Validate LHS is a single token
    if len(tokens) > 2 and tokens[1] != '=':
        raise alg_formlula_types.InvalidLHSFormulaError(
            f"'{formula}'. Left-hand side (LHS) must be a single account type."
        )

    # Validate first token is an account type
    if (
        tokens[0] in alg_formlula_types.SupportedOperationsEnum._value2member_map_
        or tokens[0] in alg_formlula_types.SpecialOperationsEnum._value2member_map_
        or tokens[0].isdigit()
        or tokens[0].replace('.', '', 1).isdigit()
    ):
        raise alg_formlula_types.InvalidOrderingError(
            f"'{formula}'. First token must be an account type, not an operator."
        )
    if tokens[0].isdigit() or tokens[0].replace('.', '', 1).isdigit():
        raise alg_formlula_types.InvalidOrderingError(
            f"'{formula}'. First token must be an account type, not a number."
        )

    # Validate only one equals sign
    n_equals = 0
    for token in tokens[2:]:
        if token == '=':
            n_equals += 1
    if n_equals > 0:
        raise alg_formlula_types.MultipleEqualsError(
            f"'{formula}'. Formula cannot contain multiple '=' operators."
        )

    # Parse LHS and RHS tokens
    lhs = AccountType(tokens[0])
    rhs_tokens = tokens[2:]

    rhs_lst: list[AccountType | alg_formlula_types.SupportedOperationsEnum | float] = []
    for token in rhs_tokens:
        if token in alg_formlula_types.SupportedOperationsEnum._value2member_map_:
            rhs_lst.append(alg_formlula_types.SupportedOperationsEnum(token))
        elif token.isdigit() or token.replace('.', '', 1).isdigit():
            rhs_lst.append(float(token))
        else:
            rhs_lst.append(AccountType(token))

    last_token_type: type = alg_formlula_types.SupportedOperationsEnum
    for token in rhs_lst:
        current_token_type: type = type(token)
        match current_token_type:
            case alg_formlula_types.SupportedOperationsEnum:
                if last_token_type == alg_formlula_types.SupportedOperationsEnum:
                    raise alg_formlula_types.InvalidOrderingError(
                        f"'{formula}'. Cannot have two consecutive tokens of the same type."
                    )
            case builtins.float:
                if last_token_type != alg_formlula_types.SupportedOperationsEnum:
                    raise alg_formlula_types.InvalidOrderingError(
                        f"'{formula}'. Numbers must follow an operator."
                    )
            case builtins.str:
                if last_token_type != alg_formlula_types.SupportedOperationsEnum:
                    raise alg_formlula_types.InvalidOrderingError(
                        f"'{formula}'. Cannot have two consecutive tokens of the same type."
                    )
            case _:
                raise alg_formlula_types.InvalidOrderingError(
                    f"'{formula}'. Unrecognized token type: {current_token_type}."
                )
        last_token_type = current_token_type

    if last_token_type == alg_formlula_types.SupportedOperationsEnum:
        raise alg_formlula_types.InvalidOrderingError(
            f"'{formula}'. Formula cannot end with an operator."
        )

    return alg_formlula_types.AlgebraicAccountFormula(lhs=lhs, rhs=rhs_lst)


def verify_formula_accounts(
    parsed_formula: alg_formlula_types.AlgebraicAccountFormula,
    available_accounts: set[AccountType],
) -> None:
    for token in parsed_formula.rhs:
        if isinstance(token, alg_formlula_types.SupportedOperationsEnum):
            continue
        elif isinstance(token, float):
            continue
        else:
            if token not in available_accounts:
                raise alg_formlula_types.IncorrectAccError(
                    f"Account '{token}' not in available accounts: {available_accounts}"
                )

    return


def validate_formulas_ordered(
    availiable_accounts: set[AccountType],
    formulas: list[alg_formlula_types.AlgebraicAccountFormula],
):
    for formula in formulas:
        verify_formula_accounts(formula, availiable_accounts)
        availiable_accounts.add(formula.lhs)  # LHS account
    return
