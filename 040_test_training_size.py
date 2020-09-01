"""
This script will generate the histogram figures shown in the referenced
paper.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coalpick import cnn, baer
from coalpick.core import load_data, shuffle_data
from coalpick.plot import plot_tranferability


def mean_absolute_error(ar1, ar2):
    """Calculate the mean absolute error."""
    diff = np.abs(ar1 - ar2)
    return np.mean(diff)


def train_or_load_cnn(weights_path, layers=None, load_scsn_weights=True):
    """
    Train or load a CNN
    """
    # Load the keras models and SCSN weights
    base_weight_path = cnn_weights_path if load_scsn_weights else None
    model = cnn.load_model(cnn_structure_path, base_weight_path)
    history = None
    # Train or load model which trains all layers
    if weights_path.exists():
        model.load_weights(str(weights_path))
    else:
        history = cnn.fit(
            model,
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            layers_to_train=layers,
        )

        # Save weights
        weights_path.parent.mkdir(exist_ok=True, parents=True)
        model.save_weights(str(weights_path))
    return model, history


def _get_results_json(path):
    """
    Load the results json file from a path or create empty list if it doesn't
    exist.
    """
    # list to store results in
    if results_path.exists():
        with results_path.open() as fi:
            results = json.load(fi)
    else:
        results = []
    return results


if __name__ == "__main__":
    # ------------------------------------- CONTROL PARAMETERS ----------------------- #
    # input parameters
    data_file = Path("data.parquet")  # Path to data file
    results_path = Path("training_quantities.json")
    out_model_path = Path("temp_models") / "variable_training_size"
    data_repeat = 5  # Number of times to repeat the data
    # Define number of passes through training data. If None, allow up to 25
    # but stop after no improvements to validation are observed for 5 epochs
    training_epochs = None

    # Number of training traces to try
    training_traces = [100, 200, 300, 400, 500, 1_000, 2_000, 3_000, 5_000, 10_000]
    training_figure_path = Path("plots") / "training_traces.png"

    # cnn parameters
    cnn_structure_path = Path("models/p_json_model.json")
    cnn_weights_path = Path("models/p_scsn_weights.hdf5")

    # Store results in dataframe
    cols = [
        "dataset",
        "train_traces",
        "mean_absolute_error",
        "model",
    ]

    results = _get_results_json(results_path)

    # ------------------------------------- PREPROCESSING ---------------------------- #
    # Load input data from parquet file
    df = load_data(data_file)

    for dataset, ds_df in df.groupby(("stats", "dataset")):
        if dataset == "B":  # B has no training data
            continue

        for training_trace_number in training_traces:
            info = {"dataset": dataset, "tra/home/derrick_chambers/Downloads/README.mdin_traces": training_trace_number}
            print(f"Working on {dataset} for {training_trace_number} traces")
            model_path = out_model_path / dataset / f"{training_trace_number}"
            # path to CNN which was trained on all layers
            expected_cnn_weights_all_path = model_path / "cnn_weights_all.hdf5"
            # path to the CNN which only trained non CNN layers (last 3)
            expected_cnn_weights_last_path = model_path / "cnn_weights_last.hdf5"
            expected_cnn_no_starting_path = model_path / "cnn_no_starting_weights.hdf5"
            # Baer params path
            expected_baer_params_path = model_path / "baer_params.json"

            # Get sampling rate for this dataset
            sr = ds_df[("stats", "sampling_rate")].iloc[0]

            # Get test and training data sub sample training data
            used_in_training = ds_df[("stats", "used_for_training")]
            train_df = ds_df[used_in_training].sample(training_trace_number)
            test_df = ds_df[~used_in_training]

            # Get arrays with analyst pick shuffled +/- 50 samples and analyst pick
            X_train, y_train = shuffle_data(train_df, repeat=data_repeat)
            X_test, y_test = shuffle_data(test_df, repeat=data_repeat)

            # ------------------------------------- CNN -------------------------------------- #
            # Make predictions after (re)training all the model's layers
            cnn_all, _ = train_or_load_cnn(expected_cnn_weights_all_path)
            cnn_all_pred = cnn.predict(cnn_all, X_test)
            mae = mean_absolute_error(cnn_all_pred, y_test)
            results.append({**{"name": "cnn_all", "mean_absolute_error": mae}, **info})

            # Make predictions after (re)training non CNN layers (outer 3)
            cnn_last, _ = train_or_load_cnn(expected_cnn_weights_last_path, layers=3)
            cnn_last_pred = cnn.predict(cnn_last, X_test)
            mae = mean_absolute_error(cnn_last_pred, y_test)
            results.append({**{"name": "cnn_last", "mean_absolute_error": mae}, **info})

            # Make predictions on a CNN starting from scratch
            cnn_no_weights, _ = train_or_load_cnn(
                expected_cnn_no_starting_path, load_scsn_weights=False
            )
            cnn_empty_pred = cnn.predict(cnn_no_weights, X_test)
            mae = mean_absolute_error(cnn_empty_pred, y_test)
            results.append(
                {**{"name": "cnn_empty", "mean_absolute_error": mae}, **info}
            )
            # ------------------------------------- BAER ------------------------------------- #
            # Creating new optimized parameters for the baer picker or load
            if expected_baer_params_path.exists():
                baer_params = baer.load_params(expected_baer_params_path)
            else:
                baer_params = baer.fit(X_train, y_train, sr)
                # Save Baer params to disk
                baer.save_params(baer_params, expected_baer_params_path)

            # Make predictions with the optimized parameters
            baer_trained_pred = baer.predict(baer_params, X_test, sr)
            mae = mean_absolute_error(baer_trained_pred, y_test)
            results.append({**{"name": "baer", "mean_absolute_error": mae}, **info})

            with results_path.open(mode="w") as fi:
                json.dump(results, fi)

    # -------------------------- Plotting ---------------------------- #
    results_df = pd.DataFrame(_get_results_json(results_path))
    plot_tranferability(results_df, output_path=training_figure_path)
