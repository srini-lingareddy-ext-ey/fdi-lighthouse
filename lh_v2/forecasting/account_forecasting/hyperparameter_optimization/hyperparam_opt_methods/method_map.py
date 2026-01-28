"""
Hyperparameter optimization method mappings.

Categories:

1. Independent Methods (pre-generate all combinations):
   - GRID: Evenly-spaced samples across parameter ranges
   - RANDOM: Uniformly random samples from parameter ranges

2. Sequential Methods (iterative generation based on history):
   - BAYESIAN: TPE-based optimization that learns from previous trials

Future sequential methods could include: GENETIC, SIMULATED_ANNEALING, etc.
"""

from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    HyperparamOptMethodEnum,
)

from .grid_search import select_params_grid
from .random_search import select_params_random

# Maps independent methods to their parameter selection functions
# Used by make_param_options() to pre-generate all combinations
HYPEROPT_METHOD_MAP = {
    HyperparamOptMethodEnum.GRID: select_params_grid,
    HyperparamOptMethodEnum.RANDOM: select_params_random,
    # Note: BAYESIAN not included - handled by sequential optimization path
}


# Maps all methods to whether they can pre-generate combinations independently
# True = Independent (can parallelize), False = Sequential (history-dependent)
INDEPENDENT_HYPEROPT_METHOD_MAP = {
    HyperparamOptMethodEnum.GRID: True,
    HyperparamOptMethodEnum.RANDOM: True,
    HyperparamOptMethodEnum.BAYESIAN: False,  # Sequential
}
