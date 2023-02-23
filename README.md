# FETSGAN
FETSGAN, RCGAN, RGAN Implementation. Synthetic time series data generation.

![fig1](https://user-images.githubusercontent.com/74554907/220963130-e34f5823-bf2d-4f2b-9117-7343601124cb.PNG)

## Data
As of now, there is no pre-processing script provided for data. All the data provided has already been pre-processed. Custom data should be of the format: `[N_SAMPLES, MAX_TIME_LENGTH, F_dim + S_dim]`, and should be stored as .json. It is highly recommended but not required to normalize the data to the range of `[-1,1]` as part of the pre-processing procedure.

### F_dim, Z_dim, and S_dim
F_dim is the dimensionality of the synthetic data, and should always be greater than 0. S_dim is optional conditional data that informs training but is not meant to be recreated by the model, and should always be the last dimensions in the .json file structure. If the conditional data is metadata that is not time-series in nature, it should be tiled. I.e:

`s.shape= [N_samples, S_dim] => s= s.unsqueeze(1).tile([1,MAX_TIME_LENGTH,1]) => data= cat([F, s])`

Z_dim is the dimensionality of the intermediate encodings. Dimensionality of 0 implies that there is no intermediate encoding process at all.

All possible configurations are:

Z_dim > 0, S_dim > 0: Conditional FETSGAN

Z_dim > 0, S_dim = 0: FETSGAN

Z_dim = 0, S_dim > 0: RCGAN

Z_dim = 0, S_dim = 0: RGAN




## Basic Usage
The code can be run with:

`python3 main.py`

Parameters are controlled by modifying the `params.yaml` file. Not all possible options are exposed to this file. However, most hidden parameters are easily changed by modifying the default function arguments throughout the repo.
