import warnings

import lightgbm as lgbm

# import numpy as np
import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingEnum
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)


class RandomForestLightGBMRanking(AbstractRankingMethod):
    """
    Ranking method using LightGBM's Random Forest implementation.

    This class implements a driver ranking approach based on LightGBM configured
    with Random Forest boosting mode. LightGBM's RF implementation is typically
    faster and more memory-efficient than traditional Random Forest implementations
    while maintaining competitive accuracy.

    The method uses feature importance scores (gain-based) from the trained model
    to rank drivers. Higher gain values indicate features that contribute more
    to reducing loss when making splits in the trees.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Object containing information about the account and drivers.
    method_params : dr_params.RandomForestLightGBMRankingParams
        Parameters for the Random Forest LightGBM ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.RandomForestLightGBMRankingParams
        Stored parameters for the ranking method.

    Methods
    -------
    apply()
        Applies the Random Forest LightGBM ranking method and returns driver rankings.
    name()
        Returns the name identifier for this ranking method.

    Notes
    -----
    LightGBM RF configuration:
    - boosting_type='rf': Uses Random Forest mode instead of gradient boosting
    - subsample < 1.0 and subsample_freq=1: Implements bagging (bootstrap sampling)
    - colsample_bytree < 1.0: Random feature selection at each tree
    - importance_type='gain': Uses total gain of splits using the feature
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.RandomForestLightGBMRankingParams,
    ):
        assert isinstance(method_params, dr_params.RandomForestLightGBMRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.RandomForestLightGBMRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'RandomForestLightGBM'

    @staticmethod
    def get_enum() -> DriverRankingEnum:
        return DriverRankingEnum.RANDOM_FOREST_LIGHTGBM

    def apply(self) -> ArrayF:
        """
        Apply LightGBM Random Forest to rank drivers by feature importance.

        This method trains a LightGBM model in Random Forest mode on the driver data
        and extracts feature importance scores based on the total gain contributed
        by each feature across all trees in the forest.

        Returns
        -------
        ArrayF
            Array of feature importance scores for each driver. Higher values
            indicate more important drivers. The array length matches the number
            of drivers in self.info.drivers.arr.

        Notes
        -----
        The model configuration uses:
        - Random Forest boosting for ensemble tree building
        - Bagging via subsample and subsample_freq
        - Random feature selection via colsample_bytree
        - Gain-based importance for interpretability
        """
        # Prepare data: transpose drivers array to get (n_samples, n_features) shape
        X = self.info.drivers.arr.T  # Shape: (n_timesteps, n_drivers)
        y = self.info.account.arr  # Shape: (n_timesteps,)

        # Create and configure LightGBM Random Forest model
        model = lgbm.LGBMRegressor(
            boosting_type='rf',  # Use Random Forest mode
            min_child_samples=self.method_params.min_child_samples,  # Min samples in leaf
            n_estimators=self.method_params.n_estimators,  # Number of trees
            subsample=self.method_params.subsample,  # Row sampling ratio for bagging
            subsample_freq=1,  # Sample at every iteration (standard RF)
            colsample_bytree=self.method_params.colsample_bytree,  # Column sampling ratio
            random_state=self.method_params.random_state,  # For reproducibility
            importance_type='gain',  # Use gain-based feature importance
            verbose=-1,  # Suppress training output
        )

        # Train the model
        model.fit(X, y)

        # Extract feature importances
        # LightGBM returns importance scores in the same order as features
        importances = model.feature_importances_

        # Convert to the expected numpy array type
        return importances.astype(self.info.np_dtype)
