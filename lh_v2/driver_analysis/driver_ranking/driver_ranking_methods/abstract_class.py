from abc import ABC, abstractmethod
from typing import Callable

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from ..driver_ranking_util import order_drivers_allow_ties, rank_array_to_ranking_dict


def error_norm(error: ArrayF, norm: Callable[[ArrayF], float] = stats.l2_norm):
    """
    Applies the given norm function to the given errors.

    Parameters
    ----------
    error : np.ndarray
        The array of error values.
    norm : function, optional
        The norm function to apply to the error array. Defaults to stats.l2_norm.

    Returns
    -------
    float
        The calculated norm of the error array.
    """

    return norm(error)


class AbstractRankingMethod(ABC):
    """
    An abstract base class defining the interface for driver ranking methods.

    This abstract class serves as a template for implementing different statistical
    methods for ranking drivers. It provides a common interface for initializing,
    applying, and ranking drivers based on various statistical techniques.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Contains information about the account and drivers, including their data arrays and mappings.
    method_params : dr_params.BaseRankingParams
        Configuration parameters for the specific ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        The account and driver information used for ranking.
    method_params : dr_params.RankingParams subclass
        The parameters for the specific ranking method.

    Methods
    -------
    apply() -> np.ndarray
        Abstract method to apply the statistical technique and return numerical outcomes.
    rank() -> dict[str, int]
        Ranks the drivers based on the outcomes of the statistical technique.
    _get_vals() -> dict[str, float]
        Returns a dictionary mapping driver names to their numerical values from the apply method.
    name() -> str
        Static method to return the name of the ranking method.
    """

    def __init__(
        self,
        account_driver_info: dts.AccountDriverGroup,
        method_params: dr_params.BaseRankingParams,
    ):
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @abstractmethod
    def apply(self) -> ArrayF:
        """
        Abstract method to apply the statistical technique and return numerical outcomes.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the numerical outcomes of the applied statistical technique.
        """
        raise NotImplementedError

    def rank(self) -> dict[dts.DriverName, int]:
        """
        Ranks the drivers based on the outcomes of the statistical technique.

        Returns
        -------
        dict[str, int]
            A dictionary mapping driver names to their ranking positions.
        """
        return rank_array_to_ranking_dict(
            rank_array=order_drivers_allow_ties(values=self.apply()),
            ordered_drivers=self.info.drivers.get_ordered_drivers(),
        )

    def get_vals(self) -> dict[dts.DriverName, float]:
        vals = self.apply()
        out_dict: dict[dts.DriverName, float] = {}
        for driver, ind in self.info.drivers.map.items():
            out_dict[driver] = float(vals[ind])
        return out_dict
