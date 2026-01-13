from itertools import product

import lh_v2.params.forecasting_params.account_forecasting_params as af

from .hyperparam_opt_methods import HYPEROPT_METHOD_MAP, HyperparamOptMethodEnum


def make_param_options(
    base_params: af.BaseAccountForecastParams,
    params_range: af.BaseAccountForecastParamsRange,
    n_options_per_param: int,
    method: HyperparamOptMethodEnum,
) -> list[af.BaseAccountForecastParams]:
    """
    Generate a list of parameter combinations for hyperparameter optimization.

    This function creates multiple parameter configurations by sampling values from
    specified ranges for each parameter in the base parameter object. It uses the
    specified optimization method to select parameter values and generates all
    possible combinations.

    Parameters
    ----------
    base_params : af.BaseAccountForecastParams
        The base parameter object containing default values and structure.
    params_range : af.BaseAccountForecastParamsRange
        The range object specifying min/max or distribution for each parameter.
        Must correspond to the type of base_params (e.g., if base_params is
        MyParams, params_range must be MyParamsRange).
    n_options_per_param : int
        Number of values to sample for each parameter from its range.
    method : HyperparamOptMethodEnum
        The hyperparameter optimization method to use for selecting parameter
        values from the ranges (e.g., grid search, random search).

    Returns
    -------
    list[af.BaseAccountForecastParams]
        A list of parameter objects, each representing a unique combination of
        parameter values sampled from the specified ranges.

    Raises
    ------
    AssertionError
        If base_params and params_range are not of corresponding types (i.e.,
        the params_range type name should be the base_params type name with
        'Range' appended).

    Notes
    -----
    The function performs the following steps:
    1. Validates that base_params and params_range are corresponding types
    2. Iterates through all fields in params_range to sample values
    3. Creates the cartesian product of all sampled parameter values
    4. Constructs parameter objects for each combination using base_params
    """
    # Validate that the parameter objects are matching types (e.g., MyParams and MyParamsRange)
    assert type(base_params).__name__ + 'Range' == type(params_range).__name__, (
        'base_params and params_range must be of corresponding types.'
    )

    # Track the index position of each parameter field for later reconstruction
    dict_fields_idx: dict[str, int] = {}
    # Store the sampled values for each parameter
    params_options: list[list[int] | list[float]] = []

    # Get the appropriate parameter selection function based on the optimization method
    select_params_func = HYPEROPT_METHOD_MAP[method]

    # Iterate through each field in the range object to sample parameter values
    for idx, field in enumerate(type(params_range).model_fields.keys()):
        # Extract the base parameter name by removing the '_range' suffix
        param_name = field[:-6]
        # Map parameter name to its index in the list for later use
        dict_fields_idx[param_name] = idx
        # Sample n_options_per_param values from this parameter's range
        params_options.append(
            select_params_func(
                params_range=getattr(params_range, field),
                n_params=n_options_per_param,
            )
        )

    # Generate all possible combinations of sampled parameter values (cartesian product)
    params_lst = list(product(*params_options))

    # Create parameter objects for each combination using the base_params factory method
    return base_params.make_all_opt_params(
        dict_fields_idx=dict_fields_idx,
        params_lst=params_lst,
    )
