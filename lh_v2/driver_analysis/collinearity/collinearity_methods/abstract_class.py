from abc import ABC, abstractmethod

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2.shared import ArrayF


class AbstractCollinearityMethod(ABC):
    """
    Abstract base class for collinearity analysis methods.

    This class serves as a template for different collinearity analysis implementations.
    All collinearity methods should inherit from this class and implement the required
    abstract methods.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Information about the drivers to analyze for collinearity.
    method_params : Union[CorrelationCollinearityParams, VIFCollinearityParams, KMeansCollinearityParams]
        Configuration parameters specific to the collinearity method being used.

    Attributes
    ----------
    info : dts.DriverGroup
        Stored information about the drivers.
    method_params : Union[CorrelationCollinearityParams, VIFCollinearityParams, KMeansCollinearityParams]
        Stored configuration parameters for the collinearity method.

    Methods
    -------
    name()
        Static method that returns the name of the collinearity method.
    apply()
        Method to execute the collinearity analysis.
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.BaseCollinearityParams,
    ):
        self.info = drivers_info
        self.method_params = method_params
        return

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Returns the name of the Collinearity Method.

        Returns
        -------
        str
            The name of the collinearity method.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self) -> ArrayF:
        """
        Apply the collinearity method.

        This method should be implemented by subclasses to perform specific
        collinearity analysis.

        Returns
        -------
        np.ndarray
            The result of the collinearity analysis.
        """
        raise NotImplementedError
