class InvalidFormulaError(Exception):
    """Custom exception for invalid formula parsing."""

    def __init__(self, message: str):
        message = 'Invalid formula format: ' + message
        super().__init__(message)
        return


class IncompleteFormulaError(InvalidFormulaError):
    """Custom exception for incomplete formula parsing."""

    def __init__(self, message: str):
        message = (
            '\nIncomplete formula: '
            'Formula does not have enough tokens. '
            'Must have at least 3 with the third token being =.\n' + message
        )
        super().__init__(message)
        return


class InvalidLHSFormulaError(InvalidFormulaError):
    """Custom exception for formulas with invalid LHS."""

    def __init__(self, message: str):
        message = (
            '\nInvalid formula: '
            'Left-hand side (LHS) of the formula must be a single account type.\n'
            + message
        )
        super().__init__(message)
        return


class MultipleEqualsError(InvalidFormulaError):
    """Custom exception for formulas with multiple equals signs."""

    def __init__(self, message: str):
        message = (
            '\nInvalid formula: '
            'Formula cannot contain multiple "=" operators.\n' + message
        )
        super().__init__(message)
        return


class InvalidOrderingError(InvalidFormulaError):
    """Custom exception for formulas with invalid ordering."""

    def __init__(self, message: str):
        message = (
            '\nInvalid formula: '
            'Formula tokens must alternate between account types and operators.\n'
            + message
        )
        super().__init__(message)
        return


class IncorrectAccError(InvalidFormulaError):
    """Custom exception for formulas with incorrect account types."""

    def __init__(self, message: str):
        message = (
            '\nInvalid formula: '
            'One or more account types in the formula are not in the available accounts.\n'
            + message
        )
        super().__init__(message)
        return
