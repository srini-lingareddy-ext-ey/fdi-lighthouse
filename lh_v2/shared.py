from typing import Any

import numpy as np
from numpy.typing import NDArray

type ArrayF = NDArray[np.floating[Any]]
"""np.ndarray alias for float arrays."""

BASE_NP_DTYPE: type[np.floating[Any]] = np.float32

type ArrayI = NDArray[np.integer[Any]]
