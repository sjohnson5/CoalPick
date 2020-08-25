"""
Preprocessing functions of SAMple.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np


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


def normalize(array, axis=-1):
    """
    Normalize an array along the last axis.
    """
    is_null = np.isnan(array)
    # mean = array.mean(axis=axis, keepdims=True)
    abs_max = np.abs(array).max(axis=axis, keepdims=True)
    return array / abs_max
