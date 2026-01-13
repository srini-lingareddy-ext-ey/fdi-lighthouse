from dataclasses import dataclass

from lh_v2.datatypes import (
    AccountGroupClassifiedDriverGroups,
    AccountType,
    DriverClassification,
    DriverName,
)
from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    DriverRankingLLMMetric,
)


@dataclass
class DriverAnalysisInput(AccountGroupClassifiedDriverGroups):
    """
    Input data structure for driver analysis.

    This class extends AccountGroupClassifiedDriverGroups to provide a typed
    input container for driver analysis operations. It inherits all attributes
    and methods from its parent class without adding additional functionality.

    Notes
    -----
    This class serves as a semantic type alias to improve code readability
    and type safety when passing input data to driver analysis functions.

    See Also
    --------
    AccountGroupClassifiedDriverGroups : Parent class containing the base structure
    """

    pass


@dataclass
class DriverAnalysisOutput:
    """
    Output data structure for driver analysis results.

    Attributes
    ----------
    accounts_drivers : AccountGroupClassifiedDriverGroups
        Collection of account groups with their classified driver groups.
    selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]]
        Nested dictionary mapping account types to driver classifications and their
        associated driver names that have been selected for analysis.
    lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Nested dictionary mapping account types to driver classifications and their
        associated driver names with corresponding lag values (in integer units).
    """

    selected_drivers: dict[AccountType, dict[DriverClassification, list[DriverName]]]
    lags: dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]

    def format_lags(self) -> dict[AccountType, dict[DriverName, int]]:
        """
        Format lags into a simplified structure.

        This method transforms the nested lag dictionary into a more accessible
        format by collapsing the driver classifications. The resulting structure
        maps each account type directly to its drivers and their corresponding lag values.

        Returns
        -------
        dict[AccountType, dict[DriverName, int]]
            A dictionary mapping account types to selected driver names and their lag values.
        """
        formatted_lags: dict[AccountType, dict[DriverName, int]] = {}

        for account in self.lags.keys():
            formatted_lags[account] = {}
            for classification in self.lags[account].keys():
                for driver in self.selected_drivers[account][classification]:
                    formatted_lags[account][driver] = self.lags[account][
                        classification
                    ][driver]

        return formatted_lags

    def format_classifications(
        self,
    ) -> dict[AccountType, dict[DriverName, DriverClassification]]:
        """
        Format classifications into a simplified structure.

        This method transforms the nested selected drivers dictionary into a more accessible
        format by collapsing the driver classifications. The resulting structure
        maps each account type directly to its drivers and their corresponding classifications.

        Returns
        -------
        dict[AccountType, dict[DriverName, DriverClassification]]
            A dictionary mapping account types to selected driver names and their classifications.
        """
        formatted_classifications: dict[
            AccountType, dict[DriverName, DriverClassification]
        ] = {}

        for account in self.selected_drivers.keys():
            formatted_classifications[account] = {}
            for classification in self.selected_drivers[account].keys():
                for driver in self.selected_drivers[account][classification]:
                    formatted_classifications[account][driver] = classification

        return formatted_classifications


@dataclass
class DriverAnalysisOutputLLM:
    selectable_drivers: dict[AccountType, dict[DriverClassification, list[DriverName]]]
    metrics: dict[
        AccountType,
        dict[
            DriverClassification, dict[DriverName, dict[DriverRankingLLMMetric, float]]
        ],
    ]
    lags: dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]

    def format_lags(
        self,
        selected_drivers: dict[
            AccountType, dict[DriverClassification, list[DriverName]]
        ],
    ) -> dict[AccountType, dict[DriverName, int]]:
        formatted_lags: dict[AccountType, dict[DriverName, int]] = {}

        for account in self.lags.keys():
            formatted_lags[account] = {}
            for classification in self.lags[account].keys():
                for driver in selected_drivers[account][classification]:
                    formatted_lags[account][driver] = self.lags[account][
                        classification
                    ][driver]

        return formatted_lags

    def format_classifications(
        self,
        selected_drivers: dict[
            AccountType, dict[DriverClassification, list[DriverName]]
        ],
    ) -> dict[AccountType, dict[DriverName, DriverClassification]]:
        formatted_classifications: dict[
            AccountType, dict[DriverName, DriverClassification]
        ] = {}

        for account in selected_drivers.keys():
            formatted_classifications[account] = {}
            for classification in selected_drivers[account].keys():
                for driver in selected_drivers[account][classification]:
                    formatted_classifications[account][driver] = classification

        return formatted_classifications
