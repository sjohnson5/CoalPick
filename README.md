# CoalPick

This simple package demonstrates training and evaluating both a CNN P phase picker
as well as [Baer Picker](https://docs.obspy.org/packages/autogen/obspy.signal.trigger.pk_baer.html#obspy.signal.trigger.pk_baer)
on several mining induced seismicity datasets.

The original CNN was developed by Ross et al. (2018) and this work is documented
in Johnson et al. (2020).

## Installation

Warning: We have only tested this code on Ubuntu 18.04. Using any other operating system
may or may not work.

You must first install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).

Next, clone coalpick.
```bash
git clone https://github.com/sjohnson5/CoalPick
```
and cd into the newly created CoalPick directory.
```bash
cd CoalPick
```
Now download the data from the following dataverse repo:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5DGFJB

and put the data.parquet file into the coalpick directory.


Then create a conda environment using the provided environment.yml file:
```bash
conda env create --file environment.yml
```

Activate the newly created environment:
```bash
conda activate coalpick
```

## Running a simple example

Now examine the example_code.py script. It is well documented and should
be easy to understand and modify. You can run it like so:
```bash
python example_code.py
```
It will generate some plots in the `plots` directory.

## Explanation of files

010_simple_example.py - Demonstrates training and evaluating both models on one dataset

020_make_figures.py - Makes the histogram figures in Johnson el al., 2020.

030_test_filling_method.py - examines how to fill missing data with minimal impact to CNN.

## References

Ross, Z. E., Meier, M. A., & Hauksson, E., 2018a. P wave arrival picking
and first‐motion polarity determination with deep learning. J. Geophys. Res.
Solid Earth. 123(6), 5120-5129.

Johnson, W. S., Chambers, D. J., Boltz, M. S., & Koper, K. D. 2020. Application
of a Convolutional Neural Network for Seismic Phase Picking of Mining-Induced
Seismicity. Geophys. J. International, in review.
