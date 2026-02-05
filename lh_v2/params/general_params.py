import datetime
from typing import Any

from pydantic import model_validator

from lh_v2.util import BaseParamsModel, ymd2pydate


class GeneralParams(BaseParamsModel):
    """
    Parameter model for general training, validation, and testing date configurations.

    This class defines the date ranges for training, validation, and testing periods,
    along with the forecast horizon. It includes validation to ensure proper date
    ordering across periods.

    Attributes
    ----------
    training_start_date : datetime.date
        Start date for the training period. Default is 2020-01-01.
    training_end_date : datetime.date
        End date for the training period. Default is 2023-12-01.
    validation_start_date : datetime.date
        Start date for the validation period. Default is 2024-01-01.
    validation_end_date : datetime.date
        End date for the validation period. Default is 2024-12-01.
    testing_start_date : datetime.date
        Start date for the testing period. Default is 2025-01-01.
    testing_end_date : datetime.date
        End date for the testing period. Default is 2025-12-01.
    n_forecast_horizon : int
        Number of periods to forecast ahead. Default is 12.

    Methods
    -------
    coerce_dates(data)
        Converts date strings in YYYYMMDD format to datetime.date objects.

    Notes
    -----
    The class enforces the following constraints:
    - Validation start date must be after training end date
    - Testing start date must be after validation end date

    Examples
    --------
    >>> params = GeneralParams(
    ...     training_start_date=ymd2pydate('2020-01-01'),
    ...     training_end_date=ymd2pydate('2023-12-01')
    ... )
    """

    training_start_date: datetime.date = ymd2pydate('2019-01-01')
    training_end_date: datetime.date = ymd2pydate('2022-12-01')
    validation_start_date: datetime.date = ymd2pydate('2023-01-01')
    validation_end_date: datetime.date = ymd2pydate('2023-12-01')
    testing_start_date: datetime.date = ymd2pydate('2024-01-01')
    testing_end_date: datetime.date = ymd2pydate('2024-06-01')

    @model_validator(mode='before')
    @classmethod
    def coerce_dates(cls, data: Any) -> Any:
        if isinstance(data, dict) and len(data) > 0:
            data['training_start_date'] = ymd2pydate(data['training_start_date'])
            data['training_end_date'] = ymd2pydate(data['training_end_date'])
            data['validation_start_date'] = ymd2pydate(data['validation_start_date'])
            data['validation_end_date'] = ymd2pydate(data['validation_end_date'])
            data['testing_start_date'] = ymd2pydate(data['testing_start_date'])
            data['testing_end_date'] = ymd2pydate(data['testing_end_date'])

        return data

    def __post_init__(self):
        assert self.validation_start_date > self.training_end_date, (
            'Validation start date must be after training end date.'
        )
        assert self.testing_start_date > self.validation_end_date, (
            'Testing start date must be after validation end date.'
        )
        return

    def get_training_daterange(self) -> tuple[datetime.date, datetime.date]:
        """Return the training date range as a tuple (start_date, end_date)."""
        return (self.training_start_date, self.training_end_date)

    def get_validation_daterange(self) -> tuple[datetime.date, datetime.date]:
        """Return the validation date range as a tuple (start_date, end_date)."""
        return (self.validation_start_date, self.validation_end_date)

    def get_train_val_daterange(self) -> tuple[datetime.date, datetime.date]:
        """Return the combined training and validation date range as a tuple (start_date, end_date)."""
        return (self.training_start_date, self.validation_end_date)

    def get_testing_daterange(self) -> tuple[datetime.date, datetime.date]:
        """Return the testing date range as a tuple (start_date, end_date)."""
        return (self.testing_start_date, self.testing_end_date)
