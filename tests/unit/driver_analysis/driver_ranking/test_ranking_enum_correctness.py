import pytest

import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
from lh_v2.driver_analysis.driver_ranking.driver_ranking_methods import (
    AbstractRankingMethod,
)
from lh_v2.driver_analysis.driver_ranking.driver_ranking_shared import (
    RANKING_METHOD_MAP,
)


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestRankingEnumCorrectness:
    def test_ranking_enum_correspondence_with_ranking_metric_enum(self):
        # Ensure that all DriverRankingEnum values are present in DriverRankingMetric
        for method in rdt.DriverRankingEnum:
            assert method.value in rdt.DriverRankingMetric._value2member_map_, (
                f'{method.value} from DriverRankingEnum is not present in DriverRankingMetric'
            )

    def test_ranking_enum_correspondence_with_ranking_method_classes(self):
        # Ensure that all DriverRankingEnum values have a corresponding method class
        for method in rdt.DriverRankingEnum:
            assert method in RANKING_METHOD_MAP.keys(), (
                f'{method.value} from DriverRankingEnum does not have a corresponding method class in RANKING_METHOD_MAP'
            )
            assert issubclass(RANKING_METHOD_MAP[method], AbstractRankingMethod), (
                f'The class corresponding to {method.value} in RANKING_METHOD_MAP is not a subclass of AbstractRankingMethod'
            )
            assert RANKING_METHOD_MAP[method].get_enum() == method, (
                f'The get_enum() method of the class corresponding to {method.value} does not return the correct enum value'
            )
