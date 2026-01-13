import datetime

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params
import lh_v2.stats as stats

from ..driver_analysis_types import DriverAnalysisInput  # , DriverAnalysisOutput


def apply_pca(
    account_drivers_info: dts.AccountDriverGroup, da_params: params.DriverAnalysisParams
) -> list[dict[dts.DriverName, float]]:
    drivers_norm = stats.normalize_arr(account_drivers_info.drivers.arr)
    account_norm = stats.normalize(account_drivers_info.account.arr)

    components, _ = stats.pca(drivers_norm)

    if da_params.full_pca_params.dim_method == rdt.PCADimMethods.ACCOUNT:
        dots = np.zeros(drivers_norm.shape[0])
        for k in range(drivers_norm.shape[0]):
            dots[k] = np.dot(drivers_norm[k], account_norm)

        used_dims = np.argsort(np.abs(dots))[::-1][
            da_params.n_final_drivers_per_classification :
        ]

    elif da_params.full_pca_params.dim_method == rdt.PCADimMethods.DRIVERS:
        used_dims = np.arange(da_params.n_final_drivers_per_classification)

    else:
        raise ValueError('Invalid PCA dimension reduction method specified.')

    pca_mappings: list[dict[dts.DriverName, float]] = []
    for idx_pca in used_dims:
        mapping: dict[dts.DriverName, float] = {}
        for driver in account_drivers_info.drivers.get_ordered_drivers():
            mapping[driver] = np.dot(
                account_drivers_info.drivers[driver],
                components[idx_pca],
            )
        pca_mappings.append(mapping)

    return pca_mappings


def _reconstruct_pca_from_mappings(
    drivers_info: dts.DriverGroup,
    pca_mappings: list[dict[dts.DriverName, float]],
) -> dts.DriverGroup:
    pca_arr = np.zeros(
        (len(pca_mappings), drivers_info.arr.shape[1]), dtype=drivers_info.np_dtype
    )
    pca_map: dict[dts.DriverName, int] = {}
    pca_dates: list[dict[datetime.date, int]] = []
    for _ in range(len(pca_mappings)):
        pca_dates.append(drivers_info.dates[0].copy())

    for idx_pca, mapping in enumerate(pca_mappings):
        pca_name = dts.DriverName(f'PCA_Driver_{idx_pca + 1}')
        pca_map[pca_name] = idx_pca
        for driver in mapping.keys():
            pca_arr[idx_pca] += drivers_info[driver] * mapping[driver]

    return dts.DriverGroup(
        arr=pca_arr,
        map=pca_map,
        dates=pca_dates,
        np_dtype=drivers_info.np_dtype,
    )


def _lagged_apply_pca(
    account_drivers_info: dts.AccountDriverGroup,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
    lag: int,
) -> dts.DriverGroup:
    lags: dict[dts.DriverName, int] = {
        driver: lag for driver in account_drivers_info.drivers.get_ordered_drivers()
    }

    pca_mappings = apply_pca(
        account_drivers_info=account_drivers_info.apply_daterange(
            start_date=general_params.training_start_date,
            end_date=general_params.training_end_date,
            best_lags=lags,
            b_training=True,
        ),
        da_params=da_params,
    )
    return _reconstruct_pca_from_mappings(
        drivers_info=account_drivers_info.drivers,
        pca_mappings=pca_mappings,
    )


def driver_analysis_pca_lagged(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
    lag: int,
) -> dts.AccountGroupSelectedDrivers:
    lst_account_pca_drivers: list[dts.DriverGroup] = []
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        account_drivers_info = accounts_drivers_info[account]
        lst_pca_drivers: list[dts.DriverGroup] = []
        for (
            classification
        ) in account_drivers_info.classified_drivers.get_ordered_classifications():
            lst_pca_drivers.append(
                _lagged_apply_pca(
                    account_drivers_info=account_drivers_info[classification],
                    general_params=general_params,
                    da_params=da_params,
                    lag=lag,
                )
            )
        acc_dg = lst_pca_drivers[0]
        for dg in lst_pca_drivers[1:]:
            acc_dg += dg
        lst_account_pca_drivers.append(acc_dg)

    return dts.AccountGroupSelectedDrivers(
        accounts=accounts_drivers_info.accounts,
        drivers=dts.ClassifiedDriverGroups.from_driver_group_lst(
            driver_group_lst=lst_account_pca_drivers,
            classification_groups=accounts_drivers_info.accounts.account_map,
        ),
    )


def driver_analysis_pca_optimize_lags(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
):  # -> dts.AccountGroupSelectedDrivers:
    # TODO: Implement optimization over multiple lags with PCA
    # Initial thoughts would be to throw all pca lags together and
    # rank them all, then take lag with highest average ranking?
    pass
