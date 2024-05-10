from typing import Optional

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_evaluator(
    model_dict: dict,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
        model_type:str,
    min_train_r2: float = 0.0,
    min_test_r2: float = 0.0,
    min_train_mae: float = 0.0,
    min_test_mae: float = 0.0,
    min_train_mse: float = 0.0,
    min_test_mse: float = 0.0,
    target: Optional[str] = "log_price",
) -> float:
    """Evaluate a trained regression model.

    Args:
        model: The pre-trained regression model artifact.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        min_train_r2: Minimal acceptable training R-squared value.
        min_test_r2: Minimal acceptable testing R-squared value.
        min_train_mae: Minimal acceptable training Mean Absolute Error value.
        min_test_mae: Minimal acceptable testing Mean Absolute Error value.
        min_train_mse: Minimal acceptable training Mean Squared Error value.
        min_test_mse: Minimal acceptable testing Mean Squared Error value.
        target: Name of target column in dataset.

    Returns:
        The R-squared score on the test set.
    """
    # Predictions
    model = model_dict[model_type]
    trn_preds = model.predict(dataset_trn.drop(columns=[target]))
    tst_preds = model.predict(dataset_tst.drop(columns=[target]))

    # Evaluation metrics
    trn_r2 = r2_score(dataset_trn[target], trn_preds)
    tst_r2 = r2_score(dataset_tst[target], tst_preds)
    trn_mae = mean_absolute_error(dataset_trn[target], trn_preds)
    tst_mae = mean_absolute_error(dataset_tst[target], tst_preds)
    trn_mse = mean_squared_error(dataset_trn[target], trn_preds)
    tst_mse = mean_squared_error(dataset_tst[target], tst_preds)

    # Logging evaluation metrics
    logger.info(f"Train R-squared={trn_r2:.4f}, MAE={trn_mae:.4f}, MSE={trn_mse:.4f}")
    logger.info(f"Test R-squared={tst_r2:.4f}, MAE={tst_mae:.4f}, MSE={tst_mse:.4f}")

    # Check against minimum acceptable values
    messages = []
    if trn_r2 < min_train_r2:
        messages.append(f"Train R-squared {trn_r2:.4f} is below {min_train_r2:.4f}!")
    if tst_r2 < min_test_r2:
        messages.append(f"Test R-squared {tst_r2:.4f} is below {min_test_r2:.4f}!")
    if trn_mae > min_train_mae:
        messages.append(f"Train MAE {trn_mae:.4f} is above {min_train_mae:.4f}!")
    if tst_mae > min_test_mae:
        messages.append(f"Test MAE {tst_mae:.4f} is above {min_test_mae:.4f}!")
    if trn_mse > min_train_mse:
        messages.append(f"Train MSE {trn_mse:.4f} is above {min_train_mse:.4f}!")
    if tst_mse > min_test_mse:
        messages.append(f"Test MSE {tst_mse:.4f} is above {min_test_mse:.4f}!")

    # Log warnings if any metrics do not meet the minimum criteria
    for message in messages:
        logger.warning(message)

    # Log metadata
    # log_artifact_metadata(
    #     metadata={
    #         "train_r2": float(trn_r2),
    #         "test_r2": float(tst_r2),
    #         "train_mae": float(trn_mae),
    #         "test_mae": float(tst_mae),
    #         "train_mse": float(trn_mse),
    #         "test_mse": float(tst_mse),
    #     },
    #     artifact_name="sklearn_regressor",
    # )

    return float(tst_r2)
