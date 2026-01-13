import lh_v2.datatypes as dts
from lh_v2.util import BaseParamsModel


class BaseDataLoaderParams(BaseParamsModel):
    pass


class DataLoadingParams(BaseParamsModel):
    selected_method: dts.data_loader_types.DataLoadingMethodEnum = (
        dts.data_loader_types.DataLoadingMethodEnum.OLD_CSV
    )
