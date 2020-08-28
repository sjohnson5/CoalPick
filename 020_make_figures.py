"""
This script will generate the histogram figures shown in the referenced
paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from coalpick import cnn, baer
from coalpick.core import load_data, shuffle_data, time_it
from coalpick.plot import plot_residuals, plot_waveforms, plot_training

if __name__ == "__main__":
    # ------------------------------------- CONTROL PARAMETERS ----------------------- #
    # input parameters
    data_file = Path("data.parquet")  # Path to data file
    results_path = "training_times.csv"
    datasets = ("A", "B", "C", "D", "E")  # The dataset to select, if None use all
    out_model_path = Path("temp_models")
    data_repeat = 5  # Number of times to repeat the data
    # Define number of passes through training data. If None, allow up to 25
    # but stop after no improvements to validation are observed for 5 epochs
    training_epochs = None

    # cnn parameters
    cnn_structure_path = Path("models/p_json_model.json")
    cnn_weights_path = Path("models/p_scsn_weights.hdf5")

    # Store timing information
    cols = [
        "CNN_train_time",
        "Baer_train_time",
        "train_traces",
        "test_traces",
        "CNN_epochs",
        "CNN_best_epoch",
    ]

    # Load profiling DF if it exists
    try:
        df_results = pd.read_csv(results_path, index_col=0)
    except FileExistsError:
        df_results = pd.DataFrame(index=datasets, columns=cols)

    # ------------------------------------- PREPROCESSING ---------------------------- #
    # Load input data from parquet file
    df = load_data(data_file)

    for dataset, ds_df in df.groupby(("stats", "dataset")):
        print(f"Working on {dataset}")

        expected_cnn_weights_path = out_model_path / dataset / "cnn_weights.hdf5"
        expected_baer_params_path = out_model_path / dataset / "baer_params.json"
        plot_path = Path("plots") / dataset

        # Dataset B has no training data, it uses Dataset A's weights
        if dataset == "B":
            expected_cnn_weights_path = out_model_path / "A" / "cnn_weights.hdf5"
            expected_baer_params_path = out_model_path / "A" / "baer_params.json"
            assert expected_baer_params_path.exists()
            assert expected_cnn_weights_path.exists()

        # Get sampling rate for this dataset
        sr = ds_df[("stats", "sampling_rate")].iloc[0]

        # Get test and training data (as described in paper)
        used_in_training = ds_df[("stats", "used_for_training")]
        train_df = ds_df[used_in_training]
        test_df = ds_df[~used_in_training]

        # Populate this datasets train/test numbers
        df_results.loc[dataset, "train_traces"] = len(train_df)
        df_results.loc[dataset, "test_traces"] = len(test_df)

        # Get arrays with analyst pick shuffled +/- 50 samples and analyst pick
        X_train, y_train = shuffle_data(train_df, repeat=data_repeat)
        X_test, y_test = shuffle_data(test_df, repeat=data_repeat)

        # ------------------------------------- CNN -------------------------------------- #
        # Load the keras models and SCSN weights
        model = cnn.load_model(cnn_structure_path, cnn_weights_path)

        # Make predictions before (re)training the model
        with time_it() as ti:
            cnn_scsn_pred = cnn.predict(model, X_test)
        df_results.loc[dataset, "Base_CNN_test_time"] = ti["duration"]

        # Train or load model
        if expected_cnn_weights_path.exists():
            model.load_weights(str(expected_cnn_weights_path))
        else:
            with time_it() as time_info:
                history = cnn.fit(
                    model,
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                )

            # Add into about timing, epochs, etc.
            df_results.loc[dataset, "CNN_train_time"] = time_info["duration"]
            val = history.history["val_mean_absolute_error"]
            df_results.loc[dataset, "CNN_epochs"] = len(val)
            df_results.loc[dataset, "CNN_best_epoch"] = np.argmin(val) + 1

            # Plot training info
            plot_training(history.history, plot_path / "training.png")

            # Save weights
            expected_cnn_weights_path.parent.mkdir(exist_ok=True, parents=True)
            model.save_weights(str(expected_cnn_weights_path))

        # Make predictions after (re)training the model
        with time_it() as ti:
            cnn_trained_pred = cnn.predict(model, X_test)
        df_results.loc[dataset, "trained_cnn_test_time"] = ti["duration"]

        # ------------------------------------- BAER ------------------------------------- #
        # Creating new optimized parameters for the baer picker or load
        if expected_baer_params_path.exists():
            baer_params = baer.load_params(expected_baer_params_path)
        else:
            with time_it() as info:
                baer_params = baer.fit(X_train, y_train, sr)

            # Log baer params
            df_results.loc[dataset, "Baer_train_time"] = info["duration"]

            # Save Baer params to disk
            baer.save_params(baer_params, expected_baer_params_path)

        # Make predictions with the optimized parameters
        with time_it() as ti:
            baer_trained_pred = baer.predict(baer_params, X_test, sr)
        df_results.loc[dataset, "trained_baer_test_time"] = ti["duration"]

        # ------------------------------------- PLOTTING --------------------------------- #

        # Plot residual histograms
        predictions = {
            "Base CNN": cnn_scsn_pred,
            "Trained Baer": baer_trained_pred,
            "Trained CNN": cnn_trained_pred,
        }
        plot_residuals(
            predictions,
            y_test,
            sr=sr,
            output_path=plot_path / "residual_histograms.png",
        )

        # Plot the first 5 waveforms and their picks.
        for i in range(5):
            picks = {
                "manual": y_test[i],
                "SCSN CNN": cnn_scsn_pred[i],
                "trained BAER": baer_trained_pred[i],
                "trained CNN": cnn_trained_pred[i],
            }
            path = plot_path / f"example_waveforms_{i}.png"
            plot_waveforms(X_test[i], picks, path)
    df_results.to_csv(results_path)
