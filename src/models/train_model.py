from typing import Any, Dict, Union
import os

import pathlib
import pickle

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts, log_artifact, catboost, sklearn
import hydra
from omegaconf import DictConfig


PROJECT_ROOT = pathlib.Path(".").absolute().resolve()

def read_data(path) -> pd.DataFrame:
    if isinstance(path, list):
        df1 = pd.read_parquet(path[0])
        df2 = pd.read_parquet(path[1])
        df = pd.concat([df1, df2])
    else:
        df = pd.read_parquet(path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration < 80)]

    pickup = df.lpep_pickup_datetime.dt
    dropoff = df.lpep_dropoff_datetime.dt

    df["month"] = pickup.month
    df["day_of_month"] = pickup.day
    df["day_of_week"] = pickup.dayofweek
    df["hour"] = pickup.hour

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def get_datasets(train_path, valid_path):
    df_train = read_data(train_path)
    df_valid = read_data(valid_path)

    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_valid["PU_DO"] = df_valid["PULocationID"] + "_" + df_valid["DOLocationID"]

    categorical = ["month", "day_of_month", "day_of_week", "hour"] #'PULocationID', 'DOLocationID', "PU_DO"]
    numerical = ["trip_distance"]
    target = "duration"
    selected_features = categorical + numerical

    X_train = df_train[selected_features]
    y_train = df_train[target]
    X_valid = df_valid[selected_features]
    y_valid = df_valid[target]
    return X_train, y_train, X_valid, y_valid


class TabularToDict(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame = None, y: Union[pd.DataFrame, None] = None) -> "TabularToDict":
        return self

    def transform(self, X: pd.DataFrame) -> Union[Dict[str, Any], list[dict[str, Any]]]:
        return X.to_dict(orient="records")


def train(model, X_train, y_train):
    model = make_pipeline(
        TabularToDict(),
        DictVectorizer(),
        model,
    )
    model.fit(X_train, y_train)
    return model


def make_prediction(model, X_valid):
    y_pred = model.predict(X_valid)
    return y_pred


def save_model(model, model_name):
    with (PROJECT_ROOT / "models" / f"{model_name}.bin").open("wb") as f_out:
        pickle.dump(model, f_out)


@hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    # get datasets
    if cfg.add_more_data:
        X_train, y_train, X_valid, y_valid = get_datasets([cfg.train_data, cfg.add_train_data], cfg.valid_data)
        log_param("data_path", f"train data - {cfg.train_data, cfg.add_train_data}, valid data - {cfg.valid_data}")
    else:
        X_train, y_train, X_valid, y_valid = get_datasets(cfg.train_data, cfg.valid_data)
        log_param("data_path", f"train data - {cfg.train_data}, valid data - {cfg.valid_data}")

    # instantiate model
    model = hydra.utils.instantiate(cfg.model)

    # train the model
    trained_model = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    # save the model
    model_name = str(cfg.model["_target_"]).split(".")[-1]
    save_model(trained_model, model_name)

    # make and save prediction
    y_pred = make_prediction(trained_model, X_valid)
    np.savetxt(f'src/models/predictions/{model_name}/pred-{model_name}.txt', y_pred)

    # logging
    t_m = trained_model.steps[2][1]
    if model_name == "CatBoostRegressor":
        catboost.log_model(t_m, f"models/{model_name}")
    else:
        sklearn.log_model(t_m, f"models/{model_name}")
    log_metric("RMSE", mean_squared_error(y_valid, y_pred, squared=False))
    log_metric("MAPE", mean_absolute_percentage_error(y_valid, y_pred))
    log_metric("R2", r2_score(y_valid, y_pred))
    log_artifacts(f"src/models/predictions/{model_name}")
    log_artifact(f"{PROJECT_ROOT}/models/{model_name}.bin")
    print(r2_score(y_valid, y_pred))

if __name__ == "__main__":
    main()
