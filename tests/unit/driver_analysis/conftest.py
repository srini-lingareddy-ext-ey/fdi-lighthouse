import datetime
from typing import TypeAlias

import numpy as np
import pytest

from lh_v2.datatypes import (
    AccountGroupClassifiedDriverGroups,
    AccountGroupInfo,
    AccountInfo,
    AccountType,
    ClassifiedDriverGroups,
    Driver,
    DriverClassification,
    DriverGroup,
    DriverName,
    HierarchyTree,
    LocationType,
    ProductType,
)
from lh_v2.datatypes.driver_analysis_types.collinearity_types import (
    CollinearityMethodEnum,
)
from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    DriverRankingEnum,
    DriverRankingMetric,
)
from lh_v2.driver_analysis.driver_ranking.driver_ranking_util import set_lags_to_zero
from lh_v2.params import DriverAnalysisParams, GeneralParams, OutputParams
from lh_v2.params.driver_analysis_params.ranking_params.extra_params import (
    RankingMethodsParams,
)
from lh_v2.shared import BASE_NP_DTYPE
from lh_v2.util.logging_util import add_custom_levels

LagMap: TypeAlias = dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]


@pytest.fixture(scope='session', autouse=True)
def configure_custom_logging_levels() -> None:
    add_custom_levels()


@pytest.fixture
def example_dates() -> list[datetime.date]:
    return [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 2, 1),
        datetime.date(2024, 3, 1),
        datetime.date(2024, 4, 1),
        datetime.date(2024, 5, 1),
        datetime.date(2024, 6, 1),
    ]


@pytest.fixture
def account_group_info(example_dates: list[datetime.date]) -> AccountGroupInfo:
    dates: dict[datetime.date, int] = {d: i for i, d in enumerate(example_dates)}
    segment = HierarchyTree(ProductType('segment_a'))
    region = HierarchyTree(LocationType('region_a'))

    acc_a = AccountInfo(
        arr=np.array([10, 12, 14, 16, 18, 20], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        account_type=AccountType('account_a'),
        segment_type=segment,
        region_type=region,
        np_dtype=BASE_NP_DTYPE,
    )
    acc_b = AccountInfo(
        arr=np.array([9, 11, 13, 15, 17, 19], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        account_type=AccountType('account_b'),
        segment_type=segment,
        region_type=region,
        np_dtype=BASE_NP_DTYPE,
    )
    return AccountGroupInfo.from_account_lst([acc_a, acc_b])


@pytest.fixture
def classified_driver_groups(
    example_dates: list[datetime.date],
) -> ClassifiedDriverGroups[DriverClassification]:
    dates: dict[datetime.date, int] = {d: i for i, d in enumerate(example_dates)}

    d1 = Driver(
        name=DriverName('ext_1'),
        arr=np.array([1, 2, 3, 4, 5, 6], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        np_dtype=BASE_NP_DTYPE,
    )
    d2 = Driver(
        name=DriverName('ext_2'),
        arr=np.array([2, 4, 6, 8, 10, 12], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        np_dtype=BASE_NP_DTYPE,
    )
    d3 = Driver(
        name=DriverName('int_1'),
        arr=np.array([6, 5, 4, 3, 2, 1], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        np_dtype=BASE_NP_DTYPE,
    )
    d4 = Driver(
        name=DriverName('int_2'),
        arr=np.array([1, 3, 1, 3, 1, 3], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        np_dtype=BASE_NP_DTYPE,
    )

    external = DriverGroup.from_driver_lst([d1, d2], np_dtype=BASE_NP_DTYPE)
    internal = DriverGroup.from_driver_lst([d3, d4], np_dtype=BASE_NP_DTYPE)

    classifications = {
        DriverClassification('External'): 0,
        DriverClassification('Internal'): 1,
    }
    return ClassifiedDriverGroups.from_driver_group_lst(
        driver_group_lst=[external, internal],
        classification_groups=classifications,
    )


@pytest.fixture
def sample_accounts_drivers_info(
    account_group_info: AccountGroupInfo,
    classified_driver_groups: ClassifiedDriverGroups[DriverClassification],
) -> AccountGroupClassifiedDriverGroups:
    return AccountGroupClassifiedDriverGroups(
        accounts=account_group_info,
        classified_drivers=classified_driver_groups,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def single_driver_accounts_info(
    example_dates: list[datetime.date],
) -> AccountGroupClassifiedDriverGroups:
    dates: dict[datetime.date, int] = {d: i for i, d in enumerate(example_dates)}
    segment = HierarchyTree(ProductType('segment_a'))
    region = HierarchyTree(LocationType('region_a'))

    acc = AccountInfo(
        arr=np.array([10, 20, 30, 40, 50, 60], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        account_type=AccountType('account_a'),
        segment_type=segment,
        region_type=region,
        np_dtype=BASE_NP_DTYPE,
    )
    accounts = AccountGroupInfo.from_account_lst([acc])

    driver = Driver(
        name=DriverName('driver_a'),
        arr=np.array([1, 2, 3, 4, 5, 6], dtype=BASE_NP_DTYPE),
        dates=dates.copy(),
        np_dtype=BASE_NP_DTYPE,
    )
    group = DriverGroup.from_driver_lst([driver], np_dtype=BASE_NP_DTYPE)
    classified = ClassifiedDriverGroups.from_driver_group_lst(
        driver_group_lst=[group],
        classification_groups={DriverClassification('External'): 0},
    )
    return AccountGroupClassifiedDriverGroups(
        accounts=accounts,
        classified_drivers=classified,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def basic_general_params() -> GeneralParams:
    return GeneralParams.model_validate(
        {
            'training_start_date': '2024-01-01',
            'training_end_date': '2024-04-01',
            'validation_start_date': '2024-05-01',
            'validation_end_date': '2024-05-01',
            'testing_start_date': '2024-06-01',
            'testing_end_date': '2024-06-01',
        }
    )


@pytest.fixture
def basic_output_params() -> OutputParams:
    return OutputParams(
        b_save_info=False,
        unique_prefix='',
        table_file_format='csv',
        other_file_format='json',
        save_dir_name='',
    )


@pytest.fixture
def minimal_driver_analysis_params() -> DriverAnalysisParams:
    da_params = DriverAnalysisParams()
    da_params.n_final_drivers_per_classification = 1
    da_params.lag_params.b_lag = False
    da_params.lag_params.n_max_lag = 0
    da_params.lag_params.n_top_considered = 1
    da_params.lag_params.ranking_methods = RankingMethodsParams(
        b_remove=False,
        methods_removed=(),
        b_selected=True,
        methods_selected=(DriverRankingEnum.SPEARMAN_CORRELATION,),
    )
    da_params.ranking_params.methods = RankingMethodsParams(
        b_remove=False,
        methods_removed=(),
        b_selected=True,
        methods_selected=(DriverRankingEnum.PEARSON_CORRELATION,),
    )
    da_params.ranking_params.ranking_metrics = (
        DriverRankingMetric.FINAL_RANK,
        DriverRankingMetric.AVG_RANK,
    )
    da_params.collinearity_params.b_remove = False
    da_params.collinearity_params.b_selected = True
    da_params.collinearity_params.methods_selected = (
        CollinearityMethodEnum.CORRELATION,
    )
    return da_params


@pytest.fixture
def zero_best_lags(
    sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
) -> LagMap:
    return set_lags_to_zero(accounts_drivers_info=sample_accounts_drivers_info)
