def _select_params_int_grid(
    params_range: tuple[int, int],
    n_params: int,
) -> list[int]:
    """
    Select evenly spaced integer parameter values from a given range.

    Parameters
    ----------
    params_range : tuple[int, int]
        A tuple containing the minimum and maximum values of the parameter range (inclusive).
    n_params : int
        The number of parameter values to select from the range.

    Returns
    -------
    list[int]
        A list of evenly spaced integer parameter values. If n_params is 1, returns the
        midpoint of the range. If the range contains fewer or equal values than n_params,
        returns all integers in the range. Otherwise, returns n_params evenly spaced
        integers across the range.

    Examples
    --------
    >>> _select_params_int_grid((0, 10), 5)
    [0, 2, 5, 7, 10]
    >>> _select_params_int_grid((0, 3), 1)
    [1]
    >>> _select_params_int_grid((0, 3), 5)
    [0, 1, 2, 3]
    """
    # Handle single parameter case: return the midpoint of the range
    if n_params == 1:
        return [int((params_range[0] + params_range[1]) / 2)]

    # If the range has fewer values than requested, return all integers in the range
    if params_range[1] - params_range[0] + 1 <= n_params:
        return list(range(params_range[0], params_range[1] + 1))
    else:
        # Calculate step size for evenly spaced values
        step = (params_range[1] - params_range[0]) / (n_params - 1)
        # Generate n_params evenly spaced integers across the range
        return [int(params_range[0] + i * step) for i in range(n_params)]


def _select_params_float_grid(
    params_range: tuple[float, float],
    n_params: int,
) -> list[float]:
    """
    Select evenly spaced float parameter values from a given range.

    Parameters
    ----------
    params_range : tuple[float, float]
        A tuple containing the minimum and maximum values of the parameter range (inclusive).
    n_params : int
        The number of parameter values to select from the range.

    Returns
    -------
    list[float]
        A list of evenly spaced float parameter values. If n_params is 1, returns the
        midpoint of the range. Otherwise, returns n_params evenly spaced floats across
        the range.

    Examples
    --------
    >>> _select_params_float_grid((0.0, 1.0), 5)
    [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> _select_params_float_grid((0.0, 1.0), 1)
    [0.5]
    """
    # Handle single parameter case: return the midpoint of the range
    if n_params == 1:
        return [(params_range[0] + params_range[1]) / 2]
    else:
        # Calculate step size for evenly spaced values
        step = (params_range[1] - params_range[0]) / (n_params - 1)
        # Generate n_params evenly spaced floats across the range
        return [params_range[0] + i * step for i in range(n_params)]


def select_params_grid(
    params_range: tuple[int, int] | tuple[float, float],
    n_params: int,
) -> list[int] | list[float]:
    """
    Select evenly spaced parameter values from a given range.

    This function dispatches to either integer or float grid selection based on the
    type of values in params_range.

    Parameters
    ----------
    params_range : tuple[int, int] or tuple[float, float]
        A tuple containing the minimum and maximum values of the parameter range (inclusive).
        Both elements must be of the same type (either both int or both float).
    n_params : int
        The number of parameter values to select from the range.

    Returns
    -------
    list[int] or list[float]
        A list of evenly spaced parameter values. Returns list[int] if params_range
        contains integers, list[float] if params_range contains floats.

    Raises
    ------
    ValueError
        If params_range contains mixed types or types other than int or float.

    Examples
    --------
    >>> select_params_grid((0, 10), 5)
    [0, 2, 5, 7, 10]
    >>> select_params_grid((0.0, 1.0), 5)
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    # Check if both elements are integers and dispatch to integer grid function
    if isinstance(params_range[0], int) and isinstance(params_range[1], int):
        return _select_params_int_grid(
            params_range=(int(params_range[0]), int(params_range[1])), n_params=n_params
        )
    # Check if both elements are floats and dispatch to float grid function
    elif isinstance(params_range[0], float) and isinstance(params_range[1], float):
        return _select_params_float_grid(params_range=params_range, n_params=n_params)
    # Raise error if params_range contains mixed types or unsupported types
    else:
        raise ValueError('params_range must be a tuple of either int or float types.')
