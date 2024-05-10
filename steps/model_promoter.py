from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


from steps.data_splitter import data_splitter
from steps.data_loader import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.model_trainer import model_trainer
from steps.model_evaluator import model_evaluator



@step
def model_promoter(r2_score: float, stage: str = "production") -> bool:
    """Model promoter step for regression models.

    This step conditionally promotes a model based on the R-squared score.
    If the R-squared score is above 0.8, the model is promoted to the specified
    stage. If there is already a model in the specified stage, the model with
    the higher R-squared score is promoted.

    Args:
        r2_score: R-squared score of the model.
        stage: Stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    is_promoted = False

    if r2_score < 0.8:
        logger.info(f"Model R-squared {r2_score:.4f} is below 0.8! Not promoting model.")
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True

        # Get the model in the current context
        current_model = get_step_context().model

        # Get the model that is in the production stage
        client = Client()
        try:
            stage_model = client.get_model_version(current_model.name, stage)
            # We compare their metrics
            prod_r2_score = (
                stage_model.get_artifact("sklearn_regressor")
                .run_metadata["test_r2"]
                .value
            )
            if float(r2_score) > float(prod_r2_score):
                # If current model has better metrics, we promote it
                is_promoted = True
                current_model.set_stage(stage, force=True)
        except KeyError:
            # If no such model exists, current one is promoted
            is_promoted = True
            current_model.set_stage(stage, force=True)

    return is_promoted

