import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params


def select_methods(
    method_params: params.AccountForecastMethodParams,
) -> list[aft.AccountForecastingMethodEnum]:
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
