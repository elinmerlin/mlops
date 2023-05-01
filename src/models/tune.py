from typing import Any, Dict, Union
import os

import pathlib
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import mlflow
from mlflow import log_metric, log_params, log_artifacts, log_artifact
import hydra
from omegaconf import DictConfig
import optuna

from train_model import get_datasets


def tune(
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_trials: int = 10,
    n_jobs: int = 1,
    tracking_uri = None,
):
    kf = KFold(n_splits=5, shuffle=True, random_state=30)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            cat_features=["PU_DO"],
            n_estimators=trial.suggest_int("n_estimators", low=50, high=500),
            max_depth=trial.suggest_int("max_depth", low=5, high=12),
            random_state=30,
        )
        metrics = []
        for train_idx, test_idx in kf.split(x, y):
            X_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = x.iloc[test_idx], y.iloc[test_idx]
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics.append(mean_squared_error(y_val, y_pred, squared=False))
        return np.round(np.mean(metrics), decimals=4)

    mlflow_callback = optuna.integration.MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="rmse",
        mlflow_kwargs={"nested": True},
    )

    study = optuna.create_study()
    study.optimize(
        func=objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[mlflow_callback],
    )
    print(study.best_params)


@hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    X_train, y_train, _, _ = get_datasets(cfg.train_data, cfg.valid_data)
    tune(x=X_train,
         y=y_train)

if __name__ == "__main__":
    main()
