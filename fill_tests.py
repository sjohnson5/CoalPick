"""
Tests for various ways of filling missing data.

Dataset C contained only event waveforms. As a result, there often wasn't
enough pre-event noise to fill the full 400 sample window required by the
CNN. This script explores the effect of repeating pre-event noise vs not doing
so on dataset A (which has no missing data issues).
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CoalPick.cnn import (
    fit,
    predict,
    load_data,
    load_model,
    shuffle_data,
    train_test_split,
)
from CoalPick.plot import plot_residuals, plot_waveforms


PRE_PICK_SAMPLES_TO_KEEP = (50, 100, 150)


def plot_fill_results(df, *, cmap='viridis', save_path=None):
    """
    Plot performance of various fill methods.

    Based on scikit-learn's visualizer (https://bit.ly/3gmhYI5)

    Parameters
    ----------
    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.
    save_path
        If not None, the path to which the figure is saved.
    """

    cm = df.values.astype(float)
    labels = df.columns
    label_inds = range(len(labels))
    fig, ax = plt.subplots()

    # n_classes = cm.shape[0]
    image = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = image.cmap(0), image.cmap(256)

    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(label_inds, label_inds):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        text_cm = format(cm[i, j], '.2g')
        if cm.dtype.kind != 'f':
            text_d = format(cm[i, j], 'd')
            if len(text_d) < len(text_cm):
                text_cm = text_d

        text_[i, j] = ax.text(
            j, i, text_cm,
            ha="center", va="center",
            color=color)

    fig.colorbar(image, ax=ax)
    ax.set(xticks=label_inds,
           yticks=label_inds,
           xticklabels=labels,
           yticklabels=labels,
           ylabel="Test Data",
           xlabel="Training Data")

    # ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45)
    if save_path is not None:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(save_path)
    return


def make_filled_dfs(train_df, test_df):
    """Create dataframes which are filled """
    def _nanify(df, samples_to_keep):
        """Create a copy of the dataframe with NaNs to simulate missing data."""
        df = df.copy()
        pick_ind = df[('stats', 'pick_sample')]
        # All the pick indices should be the same
        pick_ind_unique = pick_ind.unique()
        assert len(pick_ind_unique) == 1
        to_clear = pick_ind_unique[0] - samples_to_keep
        data = df['data']
        # Fill
        data.loc[:, :to_clear] = np.NAN
        df['data'] = data
        return df

    def repeat(df):
        """Repeat data before p pick."""
        pick_ind = df[('stats', 'pick_sample')]
        data = df['data']
        first_non_nan = int(data.isnull().idxmin(axis=1).iloc[0])
        repeat_inds = slice(int(first_non_nan), pick_ind.iloc[0])
        repeat_chunks = data.values[:, repeat_inds]
        size = repeat_chunks.shape[-1]
        start = first_non_nan - size
        data_array = data.values
        while start >= 0:
            data_array[:, start: start + size] = repeat_chunks
            start -= size
        if start != -size:
            data_array[:, :start + size] = repeat_chunks[:start + size]
        df['data'] = pd.DataFrame(data_array, index=data.index, columns=data.columns)
        return df

    df = pd.concat([train_df, test_df])
    nan_df = {s: _nanify(df, s) for s in PRE_PICK_SAMPLES_TO_KEEP}
    df_with_zeros = {f'zeros_{s}': v.fillna(value=0) for s, v in nan_df.items()}
    df_with_repeated = {f'repeat_{s}': repeat(v) for s, v in nan_df.items()}
    df_with_zeros.update(df_with_repeated)
    return {
        i: (v.loc[train_df.index], v.loc[test_df.index])
        for i, v in df_with_zeros.items()
    }


def _load_or_train_base_model(df_dict):
    """Load or train the base data."""
    out = {}
    for name, (train_df, test_df) in df_dict.items():
        expected_weight_path = model_path / f"{name}.hdf5"
        # If the model has already been trained just load it
        if expected_weight_path.exists():
            model = load_model(model_structure_path, expected_weight_path)
        else:
            # else load base model and train
            model = load_model(model_structure_path, scsn_weights_path)
            # Get arrays with analyst pick shuffled +/- 50 samples and analyst pick
            X_train, y_train = shuffle_data(train_df, repeat=training_data_repeat)
            # Train model base model and save
            fit(model, X_train, y_train, epochs=training_epochs)
            expected_weight_path.parent.mkdir(exist_ok=True, parents=True)
            model.save_weights(str(expected_weight_path))
        out[name] = model
    return out


def test_fill_methods(model_dict, df_dict):
    """
    Test each model on each type of test data.
    """
    # Create cross testing matrix
    results = []
    names = list(model_dict)
    df = pd.DataFrame(index=names, columns=names)
    for model_name, model in model_dict.items():
        for df_name, (train, test) in df_dict.items():
            X_test, y_test = shuffle_data(test)
            result = predict(model, X_test)
            resid = result - y_test
            df.loc[model_name, df_name] = np.mean(np.abs(resid))
            results.append([model_name, df_name, result])
    return df


if __name__ == "__main__":
    # Script control parameters
    data_file = Path("data.parquet")  # Path to data file
    dataset = "A"  # The dataset to select, if None use all
    model_structure_path = Path("models/p_json_model.json")
    scsn_weights_path = Path("models/p_scsn_weights.hdf5")
    train_fraction = 0.75  # fraction of traces to use for training
    training_data_repeat = 3  # Number of times to repeat training data
    training_epochs = 5  # Number of passes through training dataset
    model_path = Path('models/filling')
    mean_error_path = Path('plots') / 'fill_error.png'
    test_results_path = Path('plots') / 'fill_tests.pkl'

    # Load input data from parquet file
    df = load_data(data_file, dataset)

    # Load the SCSN keras models and weights
    model = load_model(model_structure_path, scsn_weights_path)

    # Split input dataframe into training and testing
    random_state = np.random.RandomState(seed=42)  # Use reproducible random states
    train_df, test_df = train_test_split(
        df, train_fraction=train_fraction, random_state=random_state
    )
    df_dict = make_filled_dfs(train_df, test_df)
    df_dict['base'] = (train_df, test_df)

    # load/train models
    model_dict = _load_or_train_base_model(df_dict)

    if not test_results_path.exists():
        df = test_fill_methods(model_dict, df_dict)
        df.to_pickle(test_results_path)
    else:
        df = pd.read_pickle(test_results_path)



    # Plot results
    plot_fill_results(df, save_path=Path('plots') / 'fill_results.png')

    breakpoint()
    df.iloc[df_name, model_name]

