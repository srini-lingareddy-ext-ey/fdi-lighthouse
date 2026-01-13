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
    min_mse = 1e10
    best_method: aft.AccountForecastingMethodEnum | None = None
    for method in method_map.keys():
        if (
            metrics[method_map[method]][aft.AccountValidationMetricEnum.MSE_PERCENTAGE]
            < min_mse
        ):
            min_mse = metrics[method_map[method]][
                aft.AccountValidationMetricEnum.MSE_PERCENTAGE
            ]
            best_method = method

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
    min_rmse = 1e10
    best_method: aft.AccountForecastingMethodEnum | None = None
    for method in method_map.keys():
        if (
            metrics[method_map[method]][aft.AccountValidationMetricEnum.RMSE_PERCENTAGE]
            < min_rmse
        ):
            min_rmse = metrics[method_map[method]][
                aft.AccountValidationMetricEnum.RMSE_PERCENTAGE
            ]
            best_method = method

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
    min_mape = 1e10
    best_method: aft.AccountForecastingMethodEnum | None = None
    for method in method_map.keys():
        if metrics[method_map[method]][aft.AccountValidationMetricEnum.MAPE] < min_mape:
            min_mape = metrics[method_map[method]][aft.AccountValidationMetricEnum.MAPE]
            best_method = method

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
    if selection_metric == aft.AccountValidationMetricEnum.MSE_PERCENTAGE:
        return lowest_mse_perc(method_map, metrics)
    elif selection_metric == aft.AccountValidationMetricEnum.RMSE_PERCENTAGE:
        return lowest_rmse_perc(method_map, metrics)
    elif selection_metric == aft.AccountValidationMetricEnum.MAPE:
        return lowest_mape(method_map, metrics)
    else:
        raise ValueError(f'Unsupported selection metric: {selection_metric}')
