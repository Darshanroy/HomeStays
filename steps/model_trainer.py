from typing import Optional
from typing import List
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from typing_extensions import Annotated
import os
from zenml import ArtifactConfig, step
from zenml.logger import get_logger
import pickle
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from steps.data_splitter import data_splitter
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor

logger = get_logger(__name__)


@step(enable_cache=True)
def model_trainer(model_types: List[str], dataset_trn: pd.DataFrame, target: str) -> dict:
    logger.info(f"{dataset_trn.columns}")
    trained_models = {}
    # Create directory if it doesn't exist
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")

    for model_type in model_types:
        if model_type == "sgd":
            param_grid = {'alpha': [0.001, 0.01, 0.1], 'max_iter': [1000, 2000, 3000]}
            model = GridSearchCV(SGDRegressor(), param_grid, cv=3)
        elif model_type == "catboost":
            param_grid = {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
            model = GridSearchCV(CatBoostRegressor(), param_grid, cv=3)
        elif model_type == "lgbm":
            param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
            model = GridSearchCV(LGBMRegressor(), param_grid, cv=3)
        elif model_type == "xgboost":
            param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2]}
            model = GridSearchCV(XGBRegressor(), param_grid, cv=3)
        elif model_type == "lasso":
            param_grid = {'alpha': [0.001, 0.01, 0.1]}
            model = GridSearchCV(Lasso(), param_grid, cv=3)
        else:
            raise ValueError(f"Unknown model type {model_type}")

        logger.info(f"Training model {model_type} with hyperparameter tuning...")
        model.fit(
            dataset_trn.drop(columns=[target]),
            dataset_trn[target],
        )

        trained_models[model_type] = model

        # Save the trained model as a pickle file
        save_path = os.path.join("trained_models", f"{model_type}_model.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
            logger.info(f"Model {model_type} saved as {save_path}")

    return trained_models

