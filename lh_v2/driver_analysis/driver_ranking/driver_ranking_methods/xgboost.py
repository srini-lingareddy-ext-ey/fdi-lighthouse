import warnings

import numpy as np
import sklearn.model_selection as skm
import xgboost as xgb

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingEnum
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)  # ignore convergence warnings


class XGBoostRanking(AbstractRankingMethod):
    """
    A class that implements driver ranking using XGBoost (Extreme Gradient Boosting).

    This class ranks drivers based on their importance determined by XGBoost's
    feature importance scores. The ranking is computed by performing multiple iterations
    of XGBoost regression, each time splitting the data into training and testing sets,
    and then averaging the feature importance scores across iterations.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Container for account and driver data.
    params : dr_params.XGBoostRankingParams
        Parameters for the XGBoost ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.XGBoostRankingParams
        Stored parameters for the ranking method.

    Notes
    -----
    - The method performs multiple iterations, each with a different random split of data.
    - Driver importance is determined by averaging feature importance scores across iterations.
    - XGBoost can capture non-linear relationships between drivers and the account,
      unlike linear methods such as Ridge or LASSO.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.XGBoostRankingParams,
    ):
        assert isinstance(method_params, dr_params.XGBoostRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.XGBoostRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'XGBoost'

    @staticmethod
    def get_enum() -> DriverRankingEnum:
        return DriverRankingEnum.XGBOOST

    def apply(self) -> ArrayF:
        """
        Applies the XGBoost ranking method to rank drivers.

        This method uses XGBoost to determine the importance of each driver
        based on feature importance scores. It performs multiple iterations of
        XGBoost regression, each time splitting the data into training and testing sets.
        The average feature importance scores across iterations are used as the
        ranking score.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the average feature importance scores for each driver
            across multiple iterations of XGBoost regression. Higher values indicate
            higher importance.

        Notes
        -----
        - The driver data is normalized before applying XGBoost.
        - The method performs `n_iter` iterations of XGBoost regression.
        - In each iteration, the data is split into training and testing sets using
          `sklearn.model_selection.train_test_split`.
        - An XGBoost regressor is fitted to the training data, and the feature
          importance scores are extracted.
        - The average feature importance score of each driver is calculated across all iterations.
        - XGBoost's tree-based approach naturally handles non-linear relationships and
          multicollinearity better than linear regression methods.
        """
        # Normalize driver data for more stable XGBoost results
        drivers_norm = stats.normalize_arr(self.info.drivers.arr).astype(
            self.info.np_dtype
        )

        # Initialize array to store feature importance scores from each iteration
        arr_xgb = np.zeros(
            (self.method_params.n_iter, self.info.drivers.arr.shape[0]),
            dtype=self.info.np_dtype,
        )

        # Run multiple iterations of XGBoost with different random states
        for k in range(self.method_params.n_iter):
            # Split data into training and testing sets (80/20 split)
            # Transpose drivers_norm to get features as columns (sklearn expected format)
            X_train, X_test, y_train, y_test = skm.train_test_split(  # pyright: ignore[reportUnusedVariable]
                drivers_norm.T,
                self.info.account.arr,
                test_size=0.2,
                random_state=k,  # Different random state each iteration for robust averaging
            )

            # Create and fit XGBoost regressor with parameters from config
            xgb_model = xgb.XGBRegressor(
                n_estimators=self.method_params.n_estimators,
                max_depth=self.method_params.max_depth,
                learning_rate=self.method_params.learning_rate,
                subsample=self.method_params.subsample,
                colsample_bytree=self.method_params.colsample_bytree,
                random_state=k,
                verbosity=0,  # Suppress XGBoost output
            )
            xgb_model.fit(X_train, y_train)

            # Store feature importance scores for this iteration
            # XGBoost's feature_importances_ are already non-negative
            arr_xgb[k] = xgb_model.feature_importances_

        # Average feature importance scores across iterations to get final ranking scores
        # Higher values indicate more important drivers
        return arr_xgb.mean(axis=0).astype(self.info.np_dtype)
