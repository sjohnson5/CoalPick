"""
A script for running the models to pick P phases on coal mining
induced seismicity.

Sk

See Johnson et al. 2020 for more details.
"""
from pathlib import Path
from typing import Optional, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from numpy.random import RandomState

# Use seed reproducible random states
random_state = RandomState(seed=42)


def preprocess(X: np.ndarray, y: Optional[np.newaxis] = None, fill: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ Apply preprocessing to data """

    def _normalize(array, axis=-1):
        """
        Normalize an array along the last axis.
        """
        abs_max = np.abs(array).max(axis=axis, keepdims=True)
        return array / abs_max

    def _detrend(array, axis=-1):
        """
        Detrend array along last axis.
        """
        return array

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
            y_out[:len(y)] = y
        else:
            X_out = X[:-mod]
            y_out = y[:-mod]
        assert len(X_out) % 3 == 0
        return X_out, y_out

    X_3, y_out = _threeify(X, y, fill=fill)
    X_out = _normalize(_detrend(X_3))
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
    assert data_path.exists(), 'data_file not found.'
    assert data_path.suffix == '.parquet', 'File must be a parquet file.'
    df = pd.read_parquet(data_path, engine='pyarrow')
    if dataset is not None:
        df = df.loc[df['stats', 'dataset'] == dataset]
    return df


def load_model(structure_path: Path, weights_path: Optional[Path] = None) -> keras.Model:
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
    assert structure_path.suffix == '.json', "structure_file must be a '.json' file"
    with structure_path.open('rb') as fi:
        loaded_model_json = fi.read()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
    if weights_path is not None:
        weights_path = Path(weights_path)
        assert weights_path.suffix == '.hdf5', "weights_file must be a '.hdf5' file"
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
    data = df['data'].values
    stats = df['stats']
    # Get samples to split
    samples_before = offsets + array_len // 2
    samples_after = int(array_len) - samples_before
    old_pick = stats['pick_sample'].values
    inds = np.stack([old_pick - samples_before, old_pick + samples_after])
    # Slice the new segments out of the old array
    shuffled = np.array([
        data[row, start:stop]
        for row, (start, stop) in enumerate(inds.T)
    ])
    new_pick_sample = offsets + (array_len // 2)
    return shuffled, new_pick_sample


def fit(model, data, target, epochs=2, batch_size=960):
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
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.losses.huber_loss,
                  metrics=['mean_absolute_error'])

    assert batch_size % 3 == 0, 'batch size must be multiple of 3'
    X, y = preprocess(data, target)
    model.fit(X[..., np.newaxis], y, epochs=epochs, batch_size=batch_size)
    return model


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
    assert batch_size % 3 == 0, 'batch size must be multiple of 3'
    # Adding nan data to fill to a multiple of 3
    X, _ = preprocess(data, fill=True)
    ys = model.predict(X[..., np.newaxis], batch_size=batch_size)
    # Get index to drop
    to_drop = np.isnan(X).any(axis=1)
    # Remove nan rows and drop extra dimension
    return ys[~to_drop][:, 0]


def plot_sample_figure(df, results, i=0, secondary=None):
    """ plots a figure with a pick or two on it """
    plt.figure()
    plt.plot(np.array(df.iloc[i]), c='k')
    plt.axvline(results.iloc[i], c='r', label='Primary')
    if isinstance(secondary, pd.Series):
        plt.axvline(secondary.iloc[i], c='b', label='Secondary')
        plt.legend()


def train_test_split(df: pd.DataFrame, train_fraction: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training and testing dataframes.

    Parameters
    ----------
    df
        The input dataframe which should be split
    train_fraction
        The ratio of data used for training (the rest will be used for testing).
    """
    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_df = df[~df.index.isin(train_df.index)]
    return train_df, test_df


def plot_residuals(residuals, output_path):
    """
    Plot the pick time residuals (predicted - analyst) and statistics.
    """

    def _remove_outliers(residuals):
        """Remove outliers with outer fence method."""
        q_25 = np.quantile(residuals, 0.25)
        q_75 = np.quantile(residuals, 0.75)
        inter_quantile_range = q_75 - q_25
        outer_low = q_25 - 1.5 * inter_quantile_range
        outer_high = q_75 + 1.5 * inter_quantile_range
        too_high = residuals > outer_high
        too_low = residuals < outer_low
        return residuals[~(too_high | too_low)]

    def _subplot_hists(ax, clean, color='b', stats=None):
        ax.grid(True)

        # plotting hist
        n, bins, patches = ax.hist(clean, alpha=0.35, color=color)
        ax.set_ylabel('Count')

        # # plotting line
        hists, _bins = np.histogram(clean, bins=bins)
        assert np.array_equal(bins, _bins)
        xs = np.repeat(bins, 2)[1:-1]
        ys = np.repeat(hists, 2)
        ax.plot(xs, ys, color=color)

        if stats is not None:
            # getting stats
            mean = stats["mean"]
            std = stats["std"]
            q75 = stats["q75"]
            q90 = stats["q90"]

            # plot text
            l_txt = r"$\mu=$" + f"{mean:0.3f}\n" + r"$\sigma=$" + f"{std:0.3f}"
            r_txt = r"$Q_{|75|}=$" + f"{q75:0.3f}\n" + r"$Q_{|90|}=$" + f"{q90:0.3f}"
            props = dict(boxstyle="round", alpha=0.75, facecolor="white")
            ax.text(
                0.03,
                0.93,
                l_txt,
                fontsize=10,
                bbox=props,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
            )
            ax.text(
                0.975,
                0.94,
                r_txt,
                fontsize=10,
                bbox=props,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
            )

    # Use outer fence to remove outliers and get abs of residuals
    clean = _remove_outliers(residuals)
    abs_residuals = abs(residuals)

    # get stats used in paper
    stats = dict(
        std=np.std(clean),
        mean=np.mean(clean),
        q75=np.quantile(abs_residuals, 0.75),
        q90=np.quantile(abs_residuals, 0.90),
    )

    fig, ax = plt.subplots(1, 1)
    _subplot_hists(ax, clean, stats=stats)
    ax.set_xlabel('Pick Residuals (samples)')
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)


def plot_waveforms(waveform, picks, output_path=None, buffer=30):
    """Plot zoomed in waveform and the picks made by various models. """
    picks = pd.Series(picks)
    start_sample = int(max(picks.min() - buffer, 0))
    end_sample = int(min(picks.max() + buffer, len(waveform)))
    x = np.arange(start_sample, end_sample)
    y = waveform[start_sample: end_sample]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    pick_colors = ['r', 'g', 'k', 'c']
    for (name, pick), color in zip(picks.items(), pick_colors):
        ax.axvline(pick, label=name, color=color)
    ax.legend()
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)


if __name__ == '__main__':
    # Script control parameters
    data_file = Path('data.parquet')  # Path to data file
    dataset = 'A'  # The dataset to select, if None use all
    model_structure_path = Path('models/p_json_model.json')
    model_weights_path = Path('models/p_scsn_weights.hdf5')
    output_weights_path = Path('models/trained_weights.hdf5')
    plot_path = Path('plots')  # If None dont plot
    train_fraction = 0.75  # fraction of traces to use for training
    training_data_repeat = 3  # Number of times to repeat training data
    training_epochs = 1  # Number of passes through training dataset

    # Load input data from parquet file
    df = load_data(data_file, dataset)

    # Load the keras models and weights
    model = load_model(model_structure_path, model_weights_path)

    # Split input dataframe into training and testing
    train_df, test_df = train_test_split(df, train_fraction=train_fraction)

    # Get arrays with analyst pick shuffled +/- 50 samples and analyst pick
    X_train, y_train = shuffle_data(train_df, repeat=training_data_repeat)
    X_test, y_test = shuffle_data(test_df)

    # Make predictions before (re)training the model
    predict_pre_train = predict(model, X_test)

    # Train model
    fit(model, X_train, y_train, epochs=training_epochs)

    # Save weights (uncomment next line to save the weights from training)
    # model.save_weights(output_weights_path)

    # Make predictions after (re)training the model
    predict_post_train = predict(model, X_test)

    # Make plots
    if plot_path is not None:

        # First plot residuals
        plot_residuals(predict_post_train - y_test, plot_path / 'post_train_residuals.png')
        plot_residuals(predict_pre_train - y_test, plot_path / 'pre_train_residuals.png')

        # Plot the first 5 waveforms and their picks.
        for i in range(5):
            picks = {
                'manual': y_test[i],
                'SCSN model': predict_pre_train[i],
                'retrained model': predict_post_train[i],
            }
            path = plot_path / f"example_waveforms_{i}.png"
            plot_waveforms(X_test[i], picks, path)
