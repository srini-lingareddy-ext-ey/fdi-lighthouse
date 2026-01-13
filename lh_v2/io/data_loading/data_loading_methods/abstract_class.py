from abc import ABC, abstractmethod
from typing import Any

import lh_v2.datatypes as dts
import lh_v2.params as params


class AbstractDataLoader(ABC):
    def __init__(self, lh_params: params.LighthouseParams) -> None:
        self.lh_params = lh_params
        return

    @abstractmethod
    def load_account_data(self, source: Any) -> dts.AccountGroupInfo:
        pass

    @abstractmethod
    def load_driver_data(
        self, source: Any
    ) -> dts.ClassifiedDriverGroups[dts.DriverClassification]:
        pass
