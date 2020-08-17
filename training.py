import pandas as pd
from pathlib import Path
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(array):
    """ normalize an array (divides every value by the absolute maximum of the array) """
    if all(np.isnan(array)):
        return array  # if it is a np.nan array just return it
    assert not any(np.isnan(array))
    abs_max = np.abs(array).max()
    norm_array = array / abs_max
    return norm_array


def load_data(file, dataset):
    """ loads training and test data """
    file = Path(file)
    assert file.exists(), 'data_file not found.'
    assert file.suffix == '.parquet', 'File must be a parquet file.'
    df = pd.read_parquet(file, engine='pyarrow')
    df = df.loc[df['stats', 'dataset'] == dataset]
    return df['data']


def load_model(structure_file, weights_file=None):
    """ loads a base model to train """
    structure_file = Path(structure_file)
    assert structure_file.suffix == '.json', "structure_file must be a '.json' file"
    with structure_file.open('rb') as fi:
        loaded_model_json = fi.read()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
    if weights_file is not None:
        weights_file = Path(weights_file)
        assert weights_file.suffix == '.hdf5', "weights_file must be a '.hdf5' file"
        model.load_weights(weights_file)
    return model


def train_model(model, train_df, reps=3, epochs=2, batch_size=960):
    """ trains a model """
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.losses.huber_loss,
                  metrics=['mean_absolute_error'])
    train_df = pd.concat([train_df] * reps)
    train_df = train_df.sample(frac=1)

    # Since Ross's model was made/trained on 3 gpu's (https://github.com/kuza55/keras-extras/issues/7)
    # it has to be provided data in multiples of 3
    # the batch size must also be a multiple of 3
    assert batch_size % 3 == 0
    while len(train_df) % 3 != 0:
        train_df = train_df[:-1]

    xs = np.array(train_df.apply(normalize, axis=1)).reshape(len(train_df), 400, 1)
    ys = np.full(len(train_df), 200)

    # training model
    model.fit(xs, ys, epochs=epochs, batch_size=batch_size)


def test_model(model, test_df, batch_size=960):
    # adding nan data to fill to a multiple of 3
    cnt = 0
    while len(test_df) % 3 != 0:
        test_df = test_df.append(pd.Series(name='dumby'))
        cnt += 1

    xs = np.array(test_df.apply(normalize, axis=1)).reshape(len(test_df), 400, 1)
    ys = model.predict(xs, batch_size=batch_size)

    # formatting and removing nan answers
    results = pd.Series(ys.reshape(len(ys)), index=test_df.index)
    results = results.drop('dumby')
    return results


def plot_sample_figure(df, results, i=0, secondary=None):
    plt.figure()
    plt.plot(np.array(df.iloc[i]), c='k')
    plt.axvline(results.iloc[i], c='r', label='Primary')
    if type(secondary) == pd.Series:
        plt.axvline(secondary.iloc[i], c='b', label='Secondary')
        plt.legend()


if __name__ == '__main__':
    data_file = 'data.parquet'
    dataset = 'A'
    model_structure_file = 'models/p_json_model.json'
    model_weights_file = 'models/p_ross_weights.hdf5'
    train_perc = 75

    # loading
    df = load_data(data_file, dataset)
    model = load_model(model_structure_file, model_weights_file)

    df = df.sample(n=254)  # randomly downsampling to have a usable dataset
    train_df = df.sample(frac=train_perc / 100)

    if train_perc != 100:
        # testing
        test_df = df.drop(train_df.index)
        pretest_results = test_model(model, test_df)

    # training
    train_model(model, train_df)

    if train_perc != 100:
        # testing
        test_results = test_model(model, test_df)

        # plotting some test results
        for i in range(5):
            plot_sample_figure(test_df, test_results, i=i, secondary=pretest_results)
