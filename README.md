# SAMple

This is an exSAMple script for basic usage of the CNN created by Ross et al. (2018)
on MIS data mentioned in Johnson et al. (2020).

## Installation

You must first install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).5DGFJB

Next, clone SAMple.
```bash
clone https://github.com/sjohnson5/SAMples
```
and cd into the newly created SAMples directory.
```bash
cd SAMples
```
Now download the data from the following dataverse repo:
(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5DGFJB)
and put the data.parquet file into the SAMples directory.


Then create a conda environment using the provided environment.yml file:
```bash
conda env create --file environment.yml
```
Now run the example_code.py script.
```bash
python example_code.py
```
It will generate some plots in the `plots` directory.

The example_code.py script is well documented and should provide an easy
way to adopt the code to run on your own.

## References

Ross, Z. E., Meier, M. A., & Hauksson, E., 2018a. P wave arrival picking
and first‐motion polarity determination with deep learning. J. Geophys. Res.
Solid Earth. 123(6), 5120-5129.

Johnson, W. S., Chambers, D. J., Boltz, M. S., & Koper, K. D. 2020. Application
of a Convolutional Neural Network for Seismic Phase Picking of Mining-Induced
Seismicity. Geophys. J. International, in review.