# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings

import numpy as np
import mlflow
import mlflow.sklearn
import click

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--alpha", default=0.5)
@click.option("--l1_ratio", default=0.5)
def main(alpha, l1_ratio):
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    with mlflow.start_run() as active_run:
        load_data_run = mlflow.tracking.MlflowClient().get_run(mlflow.run(".", "load_data", parameters={}).run_id)
        wine_quality_csv_uri = os.path.join(load_data_run.info.artifact_uri, "wine_quality-dir/wine_quality-red.csv")
        mlflow.run(".", "train", parameters={"wine_quality_csv": wine_quality_csv_uri, "alpha": alpha, "l1_ratio": l1_ratio})

    # data = load_data()
    #
    # train(data)


if __name__ == "__main__":
    main()
