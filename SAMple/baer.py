"""
Core functions of SAMple.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from SAMple.prep_utils import normalize

def fit(data, target):
    """
    Trains the model (modifies existing model in place).

    Parameters
    ----------
    model
        A keras model.
    data
        The input X data.
    target
        The target values the model should learn to predict.
    """

    return params


def predict(params: dict, data: np.ndarray) -> np.ndarray:
    """
    Use the params to make predictions.

    Parameters
    ----------
    params
        A dictionary of the parameters required for the baer picker
    data
        The input data in the form of a numpy ndarray.
    """

    return y



def preprocess(X: np.ndarray):
    """ Apply preprocessing to data """

    # normalize
    X_out = normalize(X)
    return X_out

