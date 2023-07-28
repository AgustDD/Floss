# TS2Vec

This repository contains the official implementation for the paper [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) (AAAI-22).

## Requirements

The recommended requirements for TS2Vec are specified as follows:
* Python 3.8
* torch==1.8.1
* scipy==1.6.1
* numpy==1.19.2
* pandas==1.0.1
* scikit_learn==0.24.2
* statsmodels==0.12.2
* Bottleneck==1.3.2

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.
* [30 UEA datasets](http://www.timeseriesclassification.com) should be put into `datasets/UEA/` so that each data file can be located by `datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.
* [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) should be put into ```datasets/``` so that each data file can be located by `datasets/<dataset_name>_*.csv`.

## Usageatasets/ElectricityLoadDiagrams20112014) should be preprocessed using `datasets/preprocess_electricity.py` and placed at `datasets/electricity.csv`.
* [Yahoo dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70) should be preprocessed using `datasets/preprocess_yahoo.py` and placed at `datasets/yahoo.pkl`.
* [KPI dataset](http://test-10056879.file.myqcloud.com/10056879/test/20180524_78431960010324/KPI%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%86%B3%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86.zip) should be preprocessed using `datasets/preprocess_kpi.py` and placed at `datasets/kpi.pkl`.


To train and evaluate TS2Vec on a dataset, run the following command:

```train & evaluate

# forecasting
python doForecasting.py

# classification
python doClassification.py

# anomaly detection
python doAnomaly.py

```

## Code Example

```python
from ts2vec import TS2Vec
import datautils

# Load the ECG200 dataset from UCR archive
train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

# Train a TS2Vec model
model = TS2Vec(
    input_dims=1,
    device=0,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)

# Compute timestamp-level representations for test set
test_repr = model.encode(test_data)  # n_instances x n_timestamps x output_dims

# Compute instance-level representations for test set
test_repr = model.encode(test_data, encoding_window='full_series')  # n_instances x output_dims

# Sliding inference for test set
test_repr = model.encode(
    test_data,
    casual=True,
    sliding_length=1,
    sliding_padding=50
)  # n_instances x n_timestamps x output_dims
# (The timestamp t's representation vector is computed using the observations located in [t-50, t])
```

