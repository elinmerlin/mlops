MLops
==============================
The main objective of this project is to gain familiarity with several MLOps tools, including MLflow, DVC, Hydra, and Optuna. The project uses the NYC Taxi dataset to train ML models to predict trip duration. However, the quality of the models is not the primary focus of the project due to a lack of time.

The following tasks were performed as part of this project:

1. DVC was set up.
2. MLflow was integrated into the training script, and several metrics were logged, including the model itself with its parameters, R2, RMSE, MAPE metrics, data path, and the model predictions. All the logged data can be found in the mlruns directory. Code organized to support hydra configurations.
3. Several models, including CatBoostRegressor, LinearRegression, DecisionTreeRegressor, GradientBoostingRegressor, and RandomForestRegressor, were trained.
4. Optuna was used to optimize the parameters for the CatBoostRegressor model.
5. CatBoostRegressor and GradientBoostingRegressor models trained on January data were registered in the MLflow model registry.
6. A Flask endpoint was implemented to serve the models (flask-app.py).
7. Batch inference is supported (batch_inference.py).
8. Additional models were trained on data from other months, and their comparison was implemented (offline_compare.py).

The following commands are available:

 - To train the default model (Linear Regression) with January data for training and February data for validation:

       python3 src/models/train_model.py

 - To apply other models and datasets, the parameters can be overwritten. For example:

       python3 src/models/train_model.py model=catboost

       python3 src/models/train_model.py model=gradient_boosting train_data=data/external/green_tripdata_2021-03.parquet

 - To add additional data (April data by default) to the training dataset:

       python3 src/models/train_model.py model=catboost add_more_data=true

 - To tune the model parameters:

       python3 src/models/tune.py

The CatBoostRegressor model is hardcoded in tune.py.

 - CatBoostRegressor and GradientBoostingRegressor models trained on January data were registered in the MLflow model registry. Additionally, new models were trained on March and March-April data. To compare the performance of these models on unseen data (May data):

       python3 src/models/offline_compare.py


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── config   <- Stores hydra configurations
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_model.py      <- Train the model
    │   │   └── batch_inference.py  <- Implement batch inference
    │   │   └── offline_compare.py  <- Compare the models
    │   │   └── tune.py             <- Optimize the model parameters
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    └── flask-app.py       <- Flask endpoint to serve the model


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
