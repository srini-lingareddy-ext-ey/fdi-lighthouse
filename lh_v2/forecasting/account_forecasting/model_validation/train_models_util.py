import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params


def select_methods(
    method_params: params.AccountForecastMethodParams,
) -> list[aft.AccountForecastingMethodEnum]:
    """
    Select which account forecasting methods to use based on configuration parameters.

    This function determines which forecasting methods should be used for account
    forecasting based on the configuration in method_params. It supports three modes:
    1. Using all available forecasting methods (default)
    2. Using all methods except those specified for removal
    3. Using only explicitly selected methods

    Parameters
    ----------
    method_params : params.AccountForecastMethodParams
        Configuration parameters specifying which forecasting methods to use. Contains
        flags (b_remove, b_selected) and method lists (methods_removed, methods_selected).

    Returns
    -------
    list[aft.AccountForecastingMethodEnum]
        List of account forecasting method enums to be used.

    Raises
    ------
    ValueError
        If both b_remove and b_selected are set to True, which is an invalid
        configuration.
    ValueError
        If the configuration state is invalid (should not occur in normal operation).

    Notes
    -----
    Exactly one of the following must be true:
    - Neither b_remove nor b_selected is True (use all methods)
    - b_remove is True (use all except specified methods)
    - b_selected is True (use only specified methods)

    Examples
    --------
    Use all available forecasting methods:

    >>> params = AccountForecastMethodParams(b_remove=False, b_selected=False)
    >>> methods = select_methods(params)

    Exclude specific methods:

    >>> params = AccountForecastMethodParams(
    ...     b_remove=True,
    ...     methods_removed=[AccountForecastingMethodEnum.LINEAR_REGRESSION]
    ... )
    >>> methods = select_methods(params)

    Use only selected methods:

    >>> params = AccountForecastMethodParams(
    ...     b_selected=True,
    ...     methods_selected=[AccountForecastingMethodEnum.XGBOOST, AccountForecastingMethodEnum.PROPHET]
    ... )
    >>> methods = select_methods(params)
    """
    # Check for invalid configuration: both flags cannot be True simultaneously
    if method_params.b_remove and method_params.b_selected:
        raise ValueError(
            'b_remove and b_selected cannot both be true. Must select one or neither to be True.'
        )

    # Default case: use all available ranking methods if no filtering is specified
    if not method_params.b_remove and not method_params.b_selected:
        return [k for k in aft.AccountForecastingMethodEnum]

    # Case: exclude specific methods from the full set
    if method_params.b_remove:
        base_list = [
            k for k in aft.AccountForecastingMethodEnum
        ]  # Start with all available methods
        for method in method_params.methods_removed:
            base_list.remove(method)  # Remove each specified method
        return base_list

    # Case: include only specifically selected methods
    if method_params.b_selected:
        return list(method_params.methods_selected)

    raise ValueError(
        'Ranking General Params not configured properly, '
        'this should not happen, if it does talk to a developer.'
    )
