# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
import tempfile
import requests

from load_data import load_data
from train import train

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # with mlflow.start_run() as active_run:
    data = load_data()

    train(data)


if __name__ == "__main__":
    main()
