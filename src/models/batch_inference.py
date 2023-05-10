import os

import pandas as pd
import mlflow

from train_model import read_data


def batch_prediction(model_uri, path):
    df = read_data(path)
    features_names = ["month", "day_of_month", "day_of_week", "hour", "trip_distance"]
    target = "duration"
    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(df[features_names])
    return pred, df[target]
