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
        # Initialize output dictionary to hold simplified lag structure
        formatted_lags: dict[AccountType, dict[DriverName, int]] = {}

        # Iterate through each account type in the lag structure
        for account in self.lags.keys():
            # Initialize dictionary for this account's driver lags
            formatted_lags[account] = {}
            # Iterate through each driver classification for this account
            for classification in self.lags[account].keys():
                # Iterate through only the selected drivers for this classification
                for driver in self.selected_drivers[account][classification]:
                    # Copy the lag value for this driver to the simplified structure
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
        # Initialize output dictionary to hold simplified classification structure
        formatted_classifications: dict[
            AccountType, dict[DriverName, DriverClassification]
        ] = {}

        # Iterate through each account type in the selected drivers structure
        for account in self.selected_drivers.keys():
            # Initialize dictionary for this account's driver classifications
            formatted_classifications[account] = {}
            # Iterate through each driver classification for this account
            for classification in self.selected_drivers[account].keys():
                # Iterate through each driver in this classification
                for driver in self.selected_drivers[account][classification]:
                    # Map the driver to its classification in the simplified structure
                    formatted_classifications[account][driver] = classification

        return formatted_classifications


@dataclass
class DriverAnalysisOutputLLM:
    """
    Output data structure for LLM-based driver analysis results.

    This class contains the results of driver analysis using LLM-based ranking,
    including selectable drivers that have passed collinearity checks, their
    ranking metrics, and optimal lag values.

    Attributes
    ----------
    selectable_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]]
        Nested dictionary mapping account types to driver classifications and their
        associated driver names that are selectable (non-collinear) for LLM-based
        driver selection.
    metrics : dict[AccountType, dict[DriverClassification, dict[DriverName, dict[DriverRankingLLMMetric, float]]]]
        Nested dictionary containing LLM ranking metrics for each selectable driver,
        organized by account type, driver classification, driver name, and metric type.
    lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Nested dictionary mapping account types to driver classifications and their
        associated driver names with corresponding optimal lag values (in integer units).

    Methods
    -------
    format_lags(selected_drivers)
        Format lags for a subset of selected drivers into a simplified structure.
    format_classifications(selected_drivers)
        Format classifications for a subset of selected drivers into a simplified structure.

    Notes
    -----
    Unlike DriverAnalysisOutput, this class retains comprehensive ranking metrics
    to support downstream LLM-based driver selection processes.

    See Also
    --------
    DriverAnalysisOutput : Standard driver analysis output without LLM metrics.
    DriverRankingLLMMetric : Enum defining available LLM ranking metrics.
    """

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
        """
        Format lags for selected drivers into a simplified structure.

        This method transforms the nested lag dictionary into a more accessible
        format by collapsing the driver classifications. Only lags for drivers
        present in the selected_drivers argument are included.

        Parameters
        ----------
        selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]]
            Nested dictionary of drivers to include in the formatted output,
            organized by account type and driver classification.

        Returns
        -------
        dict[AccountType, dict[DriverName, int]]
            A dictionary mapping account types to selected driver names and their
            corresponding lag values.

        Notes
        -----
        This method allows formatting lags for a subset of selectable drivers,
        which is useful after LLM-based driver selection has been performed.

        Examples
        --------
        >>> output_llm = DriverAnalysisOutputLLM(...)
        >>> selected = {AccountType('REVENUE'): {DriverClassification.PRIMARY: ['driver1', 'driver2']}}
        >>> formatted = output_llm.format_lags(selected)
        >>> formatted
        {AccountType('REVENUE'): {'driver1': 3, 'driver2': 1}}
        """
        formatted_lags: dict[AccountType, dict[DriverName, int]] = {}

        # Iterate through accounts in the lag structure
        for account in self.lags.keys():
            formatted_lags[account] = {}
            # Iterate through classifications for this account
            for classification in self.lags[account].keys():
                # Only include lags for drivers that are in the selected_drivers set
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
        """
        Format classifications for selected drivers into a simplified structure.

        This method transforms the nested selected drivers dictionary into a more
        accessible format by inverting the structure. The resulting dictionary maps
        each driver name to its classification.

        Parameters
        ----------
        selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]]
            Nested dictionary of drivers to include in the formatted output,
            organized by account type and driver classification.

        Returns
        -------
        dict[AccountType, dict[DriverName, DriverClassification]]
            A dictionary mapping account types to selected driver names and their
            corresponding classifications.

        Notes
        -----
        This method allows formatting classifications for a subset of selectable drivers,
        which is useful after LLM-based driver selection has been performed.

        Examples
        --------
        >>> output_llm = DriverAnalysisOutputLLM(...)
        >>> selected = {AccountType('REVENUE'): {DriverClassification.PRIMARY: ['driver1', 'driver2']}}
        >>> formatted = output_llm.format_classifications(selected)
        >>> formatted
        {AccountType('REVENUE'): {'driver1': DriverClassification.PRIMARY, 'driver2': DriverClassification.PRIMARY}}
        """
        formatted_classifications: dict[
            AccountType, dict[DriverName, DriverClassification]
        ] = {}

        # Iterate through accounts in the selected_drivers structure
        for account in selected_drivers.keys():
            formatted_classifications[account] = {}
            # Iterate through classifications for this account
            for classification in selected_drivers[account].keys():
                # Map each driver to its classification
                for driver in selected_drivers[account][classification]:
                    formatted_classifications[account][driver] = classification

        return formatted_classifications
