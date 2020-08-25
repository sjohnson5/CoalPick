"""
Core functions of SAMple.
"""

from pathlib import Path
from typing import Optional, Tuple

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json


def fit(model, data, target, epochs=2, batch_size=960, validation_data=None):
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
    epochs
        The epochs for training. Each epoch represents a complete pass through
        the entire training dataset.
    """
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.losses.huber_loss,
        metrics=["mean_absolute_error"],
    )

    assert batch_size % 3 == 0, "batch size must be multiple of 3"

    X, y = preprocess(data, target)
    # if validation data is used make sure it undergoes the same pre-processing
    if validation_data:
        validation_data = preprocess(*validation_data)

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data)
    return history


def predict(model: keras.Model, data: np.ndarray, batch_size: int = 960) -> np.ndarray:
    """
    Use the model to make predictions.

    Parameters
    ----------
    model
        A keras model.
    data
        The input data in the form of a numpy ndarray.
    batch_size
        The batch size (must be a multiple of 3).
    """
    assert batch_size % 3 == 0, "batch size must be multiple of 3"
    # Adding nan data to fill to a multiple of 3
    X, _ = preprocess(data, fill=True)
    ys = model.predict(X, batch_size=batch_size)
    # Get index to drop
    to_drop = np.isnan(X).any(axis=1)
    # Remove nan rows and drop extra dimension
    return ys[~to_drop[:, 0]][:, 0]


def preprocess(
    X: np.ndarray, y: Optional[np.newaxis] = None, fill: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """ Apply preprocessing to data """

    def _normalize(array, axis=-1):
        """
        Normalize an array along the last axis.
        """
        is_null = np.isnan(array)
        # mean = array.mean(axis=axis, keepdims=True)
        abs_max = np.abs(array).max(axis=axis, keepdims=True)
        return array / abs_max

    def _threeify(X, y=None, fill=False):
        """
        Since the SCSN model was trained on 3 GPUs data must be provided in
        multiples of 3. This function will drop up to 2 rows of X and y
        arrays so their lengths are multiples of three or, if fill is True,
        fill up to 2 rows with NaN values.

        See: https://github.com/kuza55/keras-extras/issues/7
        """
        mod = len(X) % 3
        y = np.ones(len(X)) * np.NAN if y is None else y
        assert len(X) == len(y)
        # The arrays are already multiples of 3, just return
        if mod == 0:
            return X, y
        if fill:
            rows_to_add = 3 - mod if mod != 0 else 0
            shape = list(X.shape)
            shape[0] = rows_to_add
            X_out = np.concatenate([X, np.ones(shape) * np.NAN], axis=0)
            y_out = np.ones(len(y) + rows_to_add) * np.NAN
            y_out[: len(y)] = y
        else:
            X_out = X[:-mod]
            y_out = y[:-mod]
        assert len(X_out) % 3 == 0
        return X_out, y_out

    X_3, y_out = _threeify(X, y, fill=fill)
    # normalize, add extra axis for X
    X_out = _normalize(X_3)[..., np.newaxis]
    return X_out, y_out


def load_data(data_path: Path, dataset: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the data into a dataframe.

    Parameters
    ----------
    data_path
        The path to the parquet file.
    dataset
        The name of dataset to use. Must be in {A, B, C, D, E, None}.
        None means use all datasets.

    Returns
    -------
    The loaded dataframe.
    """
    data_path = Path(data_path)
    assert data_path.exists(), "data_file not found."
    assert data_path.suffix == ".parquet", "File must be a parquet file."
    df = pd.read_parquet(data_path, engine="pyarrow")
    if dataset is not None:
        df = df.loc[df["stats", "dataset"] == dataset]
    return df


def load_model(
    structure_path: Path, weights_path: Optional[Path] = None
) -> keras.Model:
    """
    Load a keras model and, optionally, its weights.

    Parameters
    ----------
    structure_path
        A path to a json file defining the network architecture.
    weights_path
        A path to a hdf5 file containing model weights.
    """
    structure_path = Path(structure_path)
    assert structure_path.suffix == ".json", "structure_file must be a '.json' file"
    with structure_path.open("rb") as fi:
        loaded_model_json = fi.read()
    model = model_from_json(loaded_model_json, custom_objects={"tf": tf})
    if weights_path is not None:
        weights_path = Path(weights_path)
        assert weights_path.suffix == ".hdf5", "weights_file must be a '.hdf5' file"
        model.load_weights(weights_path)
    return model


def shuffle_data(df, offset=50, array_len=400, repeat=1):
    """
    Shuffles the data so that the analyst pick is within 50 samples of the
    center of a 400 sample window.

    Parameters
    ----------
    df
        The input dataframe containing data and stats
    offset
        The maximum offset from the center of the array in which the analyst
        pick can be located.
    array_len
        The output array length. It is not recommended this should change.
    repeat
        The number of times to repeat the data before shuffling. This is useful
        for increasing the number of training data.
    """
    if repeat != 1:
        df = pd.concat([df] * repeat)
    # Generate random offsets
    offsets = np.random.randint(-offset, offset, len(df))
    # Split data and meta data
    data = df["data"].values
    stats = df["stats"]
    # Get samples to split
    samples_before = offsets + array_len // 2
    samples_after = int(array_len) - samples_before
    old_pick = stats["pick_sample"].values
    inds = np.stack([old_pick - samples_before, old_pick + samples_after])
    # Slice the new segments out of the old array
    shuffled = np.array(
        [data[row, start:stop] for row, (start, stop) in enumerate(inds.T)]
    )
    new_pick_sample = offsets + (array_len // 2)
    return shuffled, new_pick_sample


def train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.75,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training and testing dataframes.

    Parameters
    ----------
    df
        The input dataframe which should be split
    train_fraction
        The ratio of data used for training (the rest will be used for testing).
    random_state
        A random state to control reproducibility.
    """
    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_df = df[~df.index.isin(train_df.index)]
    return train_df, test_df
