import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft


def lowest_mse_perc(
    method_map: dict[aft.AccountForecastingMethodEnum, int],
    metrics: list[dict[aft.AccountValidationMetricEnum, float]],
) -> aft.AccountForecastingMethodEnum:
    """
    Select the forecasting method with the lowest MSE percentage.

    Parameters
    ----------
    method_map : dict[aft.AccountForecastingMethodEnum, int]
        Mapping of forecasting methods to their index in the metrics list.
    metrics : list[dict[aft.AccountValidationMetricEnum, float]]
        List of metric dictionaries for each forecasting method.

    Returns
    -------
    aft.AccountForecastingMethodEnum
        The forecasting method with the lowest MSE percentage.

    Raises
    ------
    AssertionError
        If no best method is found based on MSE%.
    """
    # Initialize minimum MSE to a very large value for comparison
    min_mse = 1e10
    # Track the best method found so far (None initially)
    best_method: aft.AccountForecastingMethodEnum | None = None

    # Iterate through all methods to find the one with lowest MSE percentage
    for method in method_map.keys():
        # Check if this method's MSE% is better than current minimum
        if (
            metrics[method_map[method]][aft.AccountValidationMetricEnum.MSE_PERCENTAGE]
            < min_mse
        ):
            # Update minimum MSE to this method's value
            min_mse = metrics[method_map[method]][
                aft.AccountValidationMetricEnum.MSE_PERCENTAGE
            ]
            # Update best method to this one
            best_method = method

    # Ensure a best method was found (should always be true if methods exist)
    assert best_method is not None, 'No best method found based on MSE%.'
    return best_method


def lowest_rmse_perc(
    method_map: dict[aft.AccountForecastingMethodEnum, int],
    metrics: list[dict[aft.AccountValidationMetricEnum, float]],
) -> aft.AccountForecastingMethodEnum:
    """
    Select the forecasting method with the lowest RMSE percentage.

    Parameters
    ----------
    method_map : dict[aft.AccountForecastingMethodEnum, int]
        Mapping of forecasting methods to their index in the metrics list.
    metrics : list[dict[aft.AccountValidationMetricEnum, float]]
        List of metric dictionaries for each forecasting method.

    Returns
    -------
    aft.AccountForecastingMethodEnum
        The forecasting method with the lowest RMSE percentage.

    Raises
    ------
    AssertionError
        If no best method is found based on RMSE%.
    """
    # Initialize minimum RMSE to a very large value for comparison
    min_rmse = 1e10
    # Track the best method found so far (None initially)
    best_method: aft.AccountForecastingMethodEnum | None = None

    # Iterate through all methods to find the one with lowest RMSE percentage
    for method in method_map.keys():
        # Check if this method's RMSE% is better than current minimum
        if (
            metrics[method_map[method]][aft.AccountValidationMetricEnum.RMSE_PERCENTAGE]
            < min_rmse
        ):
            # Update minimum RMSE to this method's value
            min_rmse = metrics[method_map[method]][
                aft.AccountValidationMetricEnum.RMSE_PERCENTAGE
            ]
            # Update best method to this one
            best_method = method

    # Ensure a best method was found (should always be true if methods exist)
    assert best_method is not None, 'No best method found based on RMSE%.'
    return best_method


def lowest_mape(
    method_map: dict[aft.AccountForecastingMethodEnum, int],
    metrics: list[dict[aft.AccountValidationMetricEnum, float]],
) -> aft.AccountForecastingMethodEnum:
    """
    Select the forecasting method with the lowest MAPE.

    Parameters
    ----------
    method_map : dict[aft.AccountForecastingMethodEnum, int]
        Mapping of forecasting methods to their index in the metrics list.
    metrics : list[dict[aft.AccountValidationMetricEnum, float]]
        List of metric dictionaries for each forecasting method.

    Returns
    -------
    aft.AccountForecastingMethodEnum
        The forecasting method with the lowest MAPE.

    Raises
    ------
    AssertionError
        If no best method is found based on MAPE.
    """
    # Initialize minimum MAPE to a very large value for comparison
    min_mape = 1e10
    # Track the best method found so far (None initially)
    best_method: aft.AccountForecastingMethodEnum | None = None

    # Iterate through all methods to find the one with lowest MAPE
    for method in method_map.keys():
        # Check if this method's MAPE is better than current minimum
        if metrics[method_map[method]][aft.AccountValidationMetricEnum.MAPE] < min_mape:
            # Update minimum MAPE to this method's value
            min_mape = metrics[method_map[method]][aft.AccountValidationMetricEnum.MAPE]
            # Update best method to this one
            best_method = method

    # Ensure a best method was found (should always be true if methods exist)
    assert best_method is not None, 'No best method found based on MAPE.'
    return best_method


def select_best_forecast_method(
    method_map: dict[aft.AccountForecastingMethodEnum, int],
    metrics: list[dict[aft.AccountValidationMetricEnum, float]],
    selection_metric: aft.AccountValidationMetricEnum,
) -> aft.AccountForecastingMethodEnum:
    """
    Select the best forecasting method based on the specified metric.

    Parameters
    ----------
    method_map : dict[aft.AccountForecastingMethodEnum, int]
        Mapping of forecasting methods to their index in the metrics list.
    metrics : list[dict[aft.AccountValidationMetricEnum, float]]
        List of metric dictionaries for each forecasting method.
    selection_metric : aft.AccountValidationMetricEnum
        The metric to use for selecting the best method (MSE%, RMSE%, or MAPE).

    Returns
    -------
    aft.AccountForecastingMethodEnum
        The forecasting method with the best (lowest) value for the selection metric.

    Raises
    ------
    ValueError
        If the selection_metric is not supported.
    """
    # Route to appropriate selection function based on the specified metric
    if selection_metric == aft.AccountValidationMetricEnum.MSE_PERCENTAGE:
        # Select method with lowest Mean Squared Error percentage
        return lowest_mse_perc(method_map, metrics)
    elif selection_metric == aft.AccountValidationMetricEnum.RMSE_PERCENTAGE:
        # Select method with lowest Root Mean Squared Error percentage
        return lowest_rmse_perc(method_map, metrics)
    elif selection_metric == aft.AccountValidationMetricEnum.MAPE:
        # Select method with lowest Mean Absolute Percentage Error
        return lowest_mape(method_map, metrics)
    else:
        # Raise error if an unsupported metric is provided
        raise ValueError(f'Unsupported selection metric: {selection_metric}')
