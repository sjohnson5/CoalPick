"""
CNN functions of CoalPick.
"""

from pathlib import Path
from typing import Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

from CoalPick.prep_utils import normalize


def fit(model, data, target, epochs=2, layers_to_train=None, batch_size=960, validation_data=None):
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
    layers_to_train
        number of layers to train (startting with the last layer), i.e. if layers_to_train is 2 the
        last 2 layers will get trained. Default is to train all layers.
    """
    # setting trainability of model layers
    if layers_to_train is not None:
        old_layer_trainability = [layer.trainable for layer in model.layers]
        assert layers_to_train < len(model.layers)
        stop = len(model.layers) - layers_to_train
        for layer in model.layers[:stop]:
            layer.trainable = False

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.losses.huber_loss,
        metrics=["mean_absolute_error"],
    )

    assert batch_size % 3 == 0, "batch size must be multiple of 3"

    X, y = _preprocess(data, target)
    # if validation data is used make sure it undergoes the same pre-processing
    if validation_data:
        validation_data = _preprocess(*validation_data)

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data)

    # setting trainability of model layers back to the way it was
    if layers_to_train is not None:
        for layer, trainability in zip(model.layers, old_layer_trainability):
            layer.trainable = trainability
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
    X, _ = _preprocess(data, fill=True)
    ys = model.predict(X, batch_size=batch_size)
    # Get index to drop
    to_drop = np.isnan(X).any(axis=1)
    # Remove nan rows and drop extra dimension
    return ys[~to_drop[:, 0]][:, 0]


def _preprocess(
        X: np.ndarray, y: Optional[np.newaxis] = None, fill: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """ Apply preprocessing to data """

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
    X_out = normalize(X_3)[..., np.newaxis]
    return X_out, y_out


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
