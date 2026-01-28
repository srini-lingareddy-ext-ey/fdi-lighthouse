import pathlib as pth

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import BASE_NP_DTYPE

from .abstract_class import AbstractDataLoader
from .polars_loader import PolarsDataLoader


class CSVOldDataLoader(AbstractDataLoader):
    def __init__(
        self,
        lh_params: params.LighthouseParams,
        np_dtype: type[np.floating] = BASE_NP_DTYPE,
    ) -> None:
        super().__init__(lh_params)
        self.polars_loader = PolarsDataLoader(lh_params=lh_params, np_dtype=np_dtype)
        return

    def load_account_data(self, source: pth.Path) -> dts.AccountGroupInfo:
        # Implementation for loading account data from old CSV format
        assert isinstance(source, pth.Path), 'Given account source is not a pth.Path.'

        return self.polars_loader.load_account_data(source=source)

    def load_driver_data(
        self, source: pth.Path
    ) -> dts.ClassifiedDriverGroups[dts.DriverClassification]:
        assert isinstance(source, pth.Path), 'Given driver source is not a pth.Path.'

        return self.polars_loader.load_driver_data(source=source)
