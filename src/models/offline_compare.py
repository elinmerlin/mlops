import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from batch_inference import batch_prediction


@hydra.main(version_base="1.3", config_path="../config", config_name="compare.yaml")
def compare(cfg: DictConfig):
    models = {
        "Catboost trained on january data": cfg.catboost_january,
        "Catboost trained on march april data": cfg.catboost_march_april,
        "Gradient boosting trained on january data": cfg.gradient_boost_january,
        "Gradient boosting trained on march data": cfg.gradient_boost_march
    }
    may_data = cfg.unseen_data
    result = {}
    for name, model in models.items():
        y_pred, y_valid = batch_prediction(model, may_data)
        r2 = r2_score(y_valid, y_pred)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_valid, y_pred)
        result[name] = [r2, rmse, mape]

    index = ["R2", "RMSE", "MAPE"]
    comparison = pd.DataFrame(result, index=index)
    print(comparison)

if __name__ == "__main__":
    compare()
