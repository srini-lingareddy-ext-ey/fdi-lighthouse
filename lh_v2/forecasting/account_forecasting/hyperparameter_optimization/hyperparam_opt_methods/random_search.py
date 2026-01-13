import random


def _select_params_int_random(
    params_range: tuple[int, int],
    n_params: int,
) -> list[int]:
    """
    Randomly select integer parameters from a given range.

    Parameters
    ----------
    params_range : tuple[int, int]
        A tuple containing the minimum and maximum values (inclusive) for the parameter range.
    n_params : int
        The number of parameters to select.

    Returns
    -------
    list[int]
        A list of randomly selected integer parameters. If n_params exceeds the number of
        possible values in the range, returns all possible values.
    """
    # Generate all possible integer values in the range (inclusive of both bounds)
    possible_values = list(range(params_range[0], params_range[1] + 1))

    # If requested number exceeds available values, return all possible values
    if n_params >= len(possible_values):
        return possible_values

    # Randomly sample n_params unique values without replacement
    return random.sample(possible_values, n_params)


def _select_params_float_random(
    params_range: tuple[float, float],
    n_params: int,
) -> list[float]:
    """
    Randomly select float parameters from a given range.

    Parameters
    ----------
    params_range : tuple[float, float]
        A tuple containing the minimum and maximum values for the parameter range.
    n_params : int
        The number of parameters to select.

    Returns
    -------
    list[float]
        A list of randomly selected float parameters.
    """
    # Initialize set to store unique random values
    selected_params = set()

    # Generate random floats until we have the requested number of parameters
    # Using a set ensures all values are unique (though collisions are extremely rare for floats)
    while len(selected_params) < n_params:
        # Generate a random float within the specified range
        param = random.uniform(params_range[0], params_range[1])
        selected_params.add(param)

    # Convert set to list for consistent return type
    return list(selected_params)


def select_params_random(
    params_range: tuple[int, int] | tuple[float, float],
    n_params: int,
) -> list[int] | list[float]:
    """
    Select random parameter values within a specified range.

    This function generates a list of random parameter values, either integers or floats,
    depending on the type of the provided range boundaries.

    Parameters
    ----------
    params_range : tuple[int, int] | tuple[float, float]
        A tuple containing the minimum and maximum values for the parameter range.
        Both elements must be of the same type (either int or float).
    n_params : int
        The number of random parameter values to generate.

    Returns
    -------
    list[int] | list[float]
        A list of randomly selected parameter values. Returns list[int] if params_range
        contains integers, or list[float] if params_range contains floats.

    Raises
    ------
    ValueError
        If params_range contains mixed types (not both int or both float).

    Examples
    --------
    >>> select_params_random((1, 10), 5)
    [3, 7, 2, 9, 5]

    >>> select_params_random((0.0, 1.0), 3)
    [0.234, 0.789, 0.456]
    """
    # Check if both bounds are integers - delegate to integer sampling
    if isinstance(params_range[0], int) and isinstance(params_range[1], int):
        return _select_params_int_random(
            params_range=(int(params_range[0]), int(params_range[1])), n_params=n_params
        )
    # Check if both bounds are floats - delegate to float sampling
    elif isinstance(params_range[0], float) and isinstance(params_range[1], float):
        return _select_params_float_random(params_range=params_range, n_params=n_params)
    # Raise error if types are mixed or neither int nor float
    else:
        raise ValueError('params_range must be a tuple of either int or float types.')
