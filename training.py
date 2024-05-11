from steps.model_promoter import model_promoter
from steps.model_trainer import model_trainer
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.data_splitter import data_splitter
from steps.model_evaluator import model_evaluator

from typing import Optional
from uuid import UUID

from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def training(
        model_type: Optional[str] = "sgd",
):
    dataframe = data_loader(random_state=42)
    encoded_df = data_preprocessor(dataframe)
    dataset_trn, dataset_tst = data_splitter(encoded_df)
    trained_models = model_trainer([model_type], dataset_trn, 'log_price')
    score = model_evaluator(trained_models, dataset_trn, dataset_tst, model_type)
    final_ans = model_promoter(score)

    if final_ans:
        logger.info(f"Model promoted, score{score}")
    else:
        logger.info("Not promoted")


if __name__ == "__main__":
    training()
