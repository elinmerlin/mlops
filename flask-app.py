import os

import pandas as pd
import mlflow
from flask import Flask, request, jsonify


model_name = "CatBoostRegressor"
RUN_ID = "6eeeed8c9c054c758d1262753fbf6ef9"
logged_model = f"runs:/{RUN_ID}/models/{model_name}"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    ride = pd.DataFrame(ride, index=[0])
    ride.lpep_pickup_datetime = pd.to_datetime(ride.lpep_pickup_datetime)
    pickup = ride.lpep_pickup_datetime.dt
    ride["month"] = pickup.month
    ride["day_of_month"] = pickup.day
    ride["day_of_week"] = pickup.dayofweek
    ride["hour"] = pickup.hour
    features_names = ["month", "day_of_month", "day_of_week", "hour", "trip_distance"]
    ride = ride[features_names]
    return ride

def predict(features):
    pred = model.predict(features)
    return pred

app = Flask("duration-prediction")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    result = {
        "duration": pred,
        "model_version": RUN_ID
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
