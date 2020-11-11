import os
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
        wine_quality_file = os.path.join(local_dir, "wine_quality-red.csv")
        print("Downloading %s to %s" % (url, wine_quality_file))
        r = requests.get(url, stream=True)
        with open(wine_quality_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        print("Uploading file: %s" % wine_quality_file)

        mlflow.log_artifact(wine_quality_file, "wine_quality-dir")

        # try:
        #     data = pd.read_csv(wine_quality_file, sep=";")
        # except Exception as e:
        #     logger.exception(
        #         "Unable to download training & test CSV, check your internet connection. Error: %s", e
        #     )
        # return data



if __name__ == "__main__":
    load_data()
