# CoalPick

This simple package demonstrates training and evaluating both a CNN P phase picker
as well as [Baer Picker](https://docs.obspy.org/packages/autogen/obspy.signal.trigger.pk_baer.html#obspy.signal.trigger.pk_baer)
on several mining induced seismicity datasets.

The original CNN was developed by Ross et al. (2018) and this work is documented
in Johnson et al. (2020).

## Installation

Warning: We have only tested this code on Ubuntu 18.04. Using any other operating system
may or may not work.

To run these examples, you must install python. Because it is what we used to generate 
this example, we strongly recommend using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).
The following instructions assume that this is the python environment you are using.

CoalPick may be downloaded using the following command:
```bash
git clone https://github.com/sjohnson5/CoalPick
```

The remainder of these instructions assume you are working within the newly cloned `CoalPick` directory.

Within a terminal or conda prompt, create a new conda environment using the provided environment.yml file:
```bash
conda env create --file environment.yml
```

Activate the newly created environment:
```bash
conda activate coalpick
```

## Data

The data for this example can be downloaded from:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5DGFJB

The data.parquet file should be placed in the CoalPick directory.

## Explanation of files

- `010_simple_example.py`: Demonstrates training and evaluating both models on one dataset
- `020_make_figures.py`: Makes the histogram figures in Johnson el al., 2020.
- `030_test_filling_method.py`: examines how to fill missing data with minimal impact to CNN.
- `040_test_training_size.py`: explores the effects of the size of the training dataset

## Running a simple example

The `010_simple_example.py` script is the main script for demonstrating the models. It is 
well documented and should be easy to understand and modify. You can run it like so:
```bash
python 010_simple_example.py
```
It will generate some plots in the `plots` directory.


## References

Ross, Z. E., Meier, M. A., & Hauksson, E., 2018a. P wave arrival picking
and first‐motion polarity determination with deep learning. J. Geophys. Res.
Solid Earth. 123(6), 5120-5129.

Johnson, W. S., Chambers, D. J., Boltz, M. S., & Koper, K. D. 2020. Application
of a Convolutional Neural Network for Seismic Phase Picking of Mining-Induced
Seismicity. Geophys. J. International, in review.
