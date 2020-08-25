"""
A script for running the models to pick P phases on coal mining
induced seismicity.

See Johnson et al. 2020 for more details.
"""
from pathlib import Path

import numpy as np

from SAMple import cnn
from SAMple.prep_utils import (
    load_data,
    shuffle_data,
    train_test_split,
)
from SAMple.plot import plot_residuals, plot_waveforms, plot_training

if __name__ == "__main__":
    # Script control parameters
    data_file = Path("data.parquet")  # Path to data file
    dataset = "A"  # The dataset to select, if None use all
    model_structure_path = Path("models/p_json_model.json")
    model_weights_path = Path("models/p_scsn_weights.hdf5")
    output_weights_path = Path("models/trained_weights.hdf5")
    plot_path = Path("plots")  # If None dont plot
    train_fraction = 0.75  # fraction of traces to use for training
    training_data_repeat = 5  # Number of times to repeat training data
    training_epochs = 25  # Number of passes through training dataset

    # Load input data from parquet file
    df = load_data(data_file, dataset)

    # Load the keras models and weights
    model = cnn.load_model(model_structure_path, model_weights_path)

    # Split input dataframe into training and testing
    random_state = np.random.RandomState(seed=42)  # Use reproducible random states
    train_df, test_df = train_test_split(
        df, train_fraction=train_fraction, random_state=random_state
    )

    # Get arrays with analyst pick shuffled +/- 50 samples and analyst pick
    X_train, y_train = shuffle_data(train_df, repeat=training_data_repeat)
    X_test, y_test = shuffle_data(test_df)

    # Make predictions before (re)training the model
    predict_pre_train = cnn.predict(model, X_test)

    # Train model
    history = cnn.fit(model, X_train, y_train, epochs=training_epochs,
                      validation_data=(X_test, y_test))

    # Save weights (uncomment next line to save the weights from training)
    # model.save_weights(output_weights_path)

    # Make predictions after (re)training the model
    predict_post_train = cnn.predict(model, X_test)

    # Make plots
    if plot_path is not None:

        # Plot training losses
        plot_training(history.history, plot_path / 'training.png')

        # Plot residual histograms
        plot_residuals(
            predict_post_train - y_test, plot_path / "post_train_residuals.png"
        )
        plot_residuals(
            predict_pre_train - y_test, plot_path / "pre_train_residuals.png"
        )

        # Plot the first 5 waveforms and their picks.
        for i in range(5):
            picks = {
                "manual": y_test[i],
                "SCSN model": predict_pre_train[i],
                "retrained model": predict_post_train[i],
            }
            path = plot_path / f"example_waveforms_{i}.png"
            plot_waveforms(X_test[i], picks, path)
