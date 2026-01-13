import time
from typing import Any

import numpy as np

import lh_v2.datatypes.data_loader_types as ldt
import lh_v2.params as params
from lh_v2.driver_analysis import DriverAnalysisInput
from lh_v2.util import get_logger

from .data_loading_methods import AbstractDataLoader, CSVOldDataLoader

logger = get_logger(__name__)

DATA_LOADING_METHOD_MAP: dict[ldt.DataLoadingMethodEnum, type[AbstractDataLoader]] = {
    ldt.DataLoadingMethodEnum.OLD_CSV: CSVOldDataLoader,
}


def load_data_driver_ranking(
    lh_params: params.LighthouseParams, account_source: Any, driver_source: Any
) -> DriverAnalysisInput:
    data_loading_method: AbstractDataLoader = DATA_LOADING_METHOD_MAP[
        lh_params.load_data_params.selected_method
    ](lh_params=lh_params)
    logger.info(
        f'Loading data using method: {lh_params.load_data_params.selected_method.value}'
    )

    start_time = time.time()
    last_time = time.time()

    accounts_info = data_loading_method.load_account_data(source=account_source)
    logger.timing(f'Account data loaded in {time.time() - last_time:.4f} seconds.')

    last_time = time.time()
    drivers_info = data_loading_method.load_driver_data(source=driver_source)
    logger.timing(f'Driver data loaded in {time.time() - last_time:.4f} seconds.')

    logger.timing(f'Total data loading time: {time.time() - start_time:.4f} seconds.')

    return DriverAnalysisInput(
        accounts=accounts_info,
        classified_drivers=drivers_info,
        np_dtype=np.float32,
    )
