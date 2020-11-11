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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def load_data():
    with mlflow.start_run():
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        local_dir = tempfile.mkdtemp()
        winequality_file = os.path.join(local_dir, "winequality-red.csv")
        print("Downloading %s to %s" % (url, winequality_file))
        r = requests.get(url, stream=True)
        with open(winequality_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        print("Uploading file: %s" % winequality_file)

        mlflow.log_artifact(winequality_file, "winequality-dir")

        # Read the wine-quality csv file from the URL
        csv_url = (
            "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        )
        try:
            data = pd.read_csv(winequality_file, sep=";")
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s", e
            )
        return data



if __name__ == "__main__":
    load_data()
