"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared fixtures available to all tests
- Test configuration
- Custom pytest hooks
"""

import pathlib as pth

import numpy as np
import pytest

from lh_v2.datatypes import DriverClassification
from lh_v2.params import LighthouseParams, parse_yaml
from lh_v2.shared import BASE_NP_DTYPE, ArrayF


@pytest.fixture(scope='session')
def project_root() -> pth.Path:
    """Return the project root directory."""
    return pth.Path(__file__).parent.parent


@pytest.fixture(scope='session')
def testing_data_dir(project_root: pth.Path) -> pth.Path:
    """Return the sample data directory."""
    return project_root / 'tests' / 'test_data'


@pytest.fixture(scope='session')
def testing_acc_data_pth(testing_data_dir: pth.Path) -> pth.Path:
    """Return the path to the test account data Parquet file."""
    return testing_data_dir / 'reduced_acc_data.parquet'


@pytest.fixture(scope='session')
def testing_driver_data_pth(testing_data_dir: pth.Path) -> pth.Path:
    """Return the path to the test driver data Parquet file."""
    return testing_data_dir / 'reduced_driver_data.parquet'


@pytest.fixture(scope='session')
def config_path(project_root: pth.Path) -> pth.Path:
    """Return the path to the config.yml file."""
    return project_root / 'lh_v2' / 'config.yml'


@pytest.fixture
def default_params() -> LighthouseParams:
    """Return default LighthouseParams for testing."""
    return LighthouseParams()


@pytest.fixture(scope='session')
def simple_params() -> LighthouseParams:
    """Return simple LighthouseParams for testing."""
    simple_param_dict = {
        'use_default': False,
        'use_default_params': True,
        'accounts': ['simple_account1', 'simple_account2'],
        'segment': 'simple_segment',
        'region': 'simple_region',
        'general_params': {
            'training_start_date': '2020-01-01',
            'training_end_date': '2020-12-01',
            'validation_start_date': '2021-01-01',
            'validation_end_date': '2021-06-01',
            'testing_start_date': '2021-07-01',
            'testing_end_date': '2021-12-01',
        },
    }
    return parse_yaml(simple_param_dict)


@pytest.fixture(scope='session')
def parsed_params(config_path: pth.Path) -> LighthouseParams:
    """Return parsed LighthouseParams from config.yml."""
    return parse_yaml(config_path)


@pytest.fixture(scope='session')
def sample_params() -> LighthouseParams:
    """Return sample LighthouseParams for testing."""
    sample_params_dict = {
        'use_default': False,
        'use_default_params': True,
        'accounts': [
            'volume',
            'net_revenue',
            'cogs_total',
            't_w_total',
            'gross_margin',
        ],
        'segment': 'Residential',
        'region': 'North America',
        'general_params': {
            'training_start_date': '2019-01-01',
            'training_end_date': '2022-12-01',
            'validation_start_date': '2023-01-01',
            'validation_end_date': '2023-12-01',
            'testing_start_date': '2024-01-01',
            'testing_end_date': '2024-06-01',
        },
    }
    return parse_yaml(sample_params_dict)


@pytest.fixture(scope='session')
def sample_driver_classifications() -> list[DriverClassification]:
    """Return sample driver classifications for testing."""
    return [DriverClassification('External'), DriverClassification('Internal')]


@pytest.fixture
def sample_array() -> ArrayF:
    """Return a sample NumPy array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def sample_time_series() -> ArrayF:
    """Return a sample time series NumPy array for testing."""
    return np.array(
        [10.0, 12.0, 13.5, 15.0, 14.0, 16.0, 18.0, 20.0, 19.0, 21.0],
        dtype=BASE_NP_DTYPE,
    )
