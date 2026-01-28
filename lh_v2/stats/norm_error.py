import numpy as np

from lh_v2.shared import ArrayF


def l1_norm(vector: ArrayF) -> float:
    """
    Calculate the L1 norm (Manhattan norm) of a vector.

    Parameters
    ----------
    vector : np.ndarray
        The input vector (shape - (n,)).

    Returns
    -------
    float
        The L1 norm of the vector.
    """
    return float(np.sum(np.abs(vector)))


def l2_norm(vector: ArrayF) -> float:
    """
    Calculate the L2 norm (Euclidean norm) of a vector.

    Parameters
    ----------
    vector : np.ndarray
        The input vector (shape - (n,)).

    Returns
    -------
    float
        The L2 norm of the vector.
    """
    return float(np.sqrt(np.sum(np.square(vector))))


def mse(model: ArrayF, true: ArrayF) -> float:
    """
    Calculate the Mean Squared Error (MSE) between the model predictions and true values.

    Parameters
    ----------
    model : np.ndarray
        The predicted values from the model (shape - (n,)).
    true : np.ndarray
        The true values (shape - (n,)).

    Returns
    -------
    float
        The Mean Squared Error between the model predictions and true values.
    """
    return float(np.sum(np.square(model - true)) / model.shape[0])


def mse_percentage(model: ArrayF, true: ArrayF) -> float:
    """
    Calculate the Mean Squared Error Percentage (MSEP)
    between the model predictions and true values.

    Parameters
    ----------
    model : np.ndarray
        The predicted values from the model (shape - (n,)).
    true : np.ndarray
        The true values (shape - (n,)).

    Returns
    -------
    float
        The Mean Squared Error Percentage between the model predictions and true values.
    """
    return float(np.sum(np.square((true - model) / true)) / model.shape[0] * 100)


def rmse(model: ArrayF, true: ArrayF) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE)
    between the model predictions and true values.

    Parameters
    ----------
    model : np.ndarray
        The predicted values from the model (shape - (n,)).
    true : np.ndarray
        The true values (shape - (n,)).

    Returns
    -------
    float
        The Root Mean Squared Error between the model predictions and true values.
    """
    return float(np.sqrt(mse(model, true)))


def rmse_percentage(model: ArrayF, true: ArrayF) -> float:
    """
    Calculate the Root Mean Squared Error Percentage (RMSEP)
    between the model predictions and true values.

    Parameters
    ----------
    model : np.ndarray
        The predicted values from the model (shape - (n,)).
    true : np.ndarray
        The true values (shape - (n,)).

    Returns
    -------
    float
        The Root Mean Squared Error Percentage between the model predictions and true values.
    """
    return float(
        np.sqrt(np.sum(np.square((true - model) / true)) / model.shape[0]) * 100
    )


def mape(model: ArrayF, true: ArrayF) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE)
    between the model predictions and true values.

    Parameters
    ----------
    model : np.ndarray
        The predicted values from the model (shape - (n,)).
    true : np.ndarray
        The true values (shape - (n,)).

    Returns
    -------
    float
        The Mean Absolute Percentage Error between the model predictions and true values.
    """
    return float(np.sum(np.abs((true - model) / true)) / model.shape[0])


def rsquared(model: ArrayF, true: ArrayF) -> float:
    """
    Calculate the R-squared (coefficient of determination)
    between the model predictions and true values.

    Parameters
    ----------
    model : np.ndarray
        The predicted values from the model (shape - (n,)).
    true : np.ndarray
        The true values (shape - (n,)).

    Returns
    -------
    float
        The R-squared value indicating the goodness of fit.
    """
    return float(
        1 - np.sum(np.square(true - model)) / np.sum(np.square(true - true.mean()))
    )
