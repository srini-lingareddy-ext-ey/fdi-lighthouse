from lh_v2.datatypes.driver_analysis_types.ranking_types import PCADimMethods
from lh_v2.util import BaseParamsModel


class FullPCAParams(BaseParamsModel):
    dim_method: PCADimMethods = PCADimMethods.ACCOUNT
