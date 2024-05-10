import pandas as pd
from typing_extensions import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_loader(
    random_state: int, is_inference: bool = False, target: str = "log_price"
) -> Annotated[pd.DataFrame, "dataset"]:


    dataset = pd.read_csv("Dataset/Homestays_Data(in).csv")
    inference_size = int(len(dataset) * 0.05)
    inference_subset = dataset.sample(
        inference_size, random_state=random_state
    )
    if is_inference:
        dataset = inference_subset
        dataset.drop(columns=target, inplace=True)
    else:
        dataset.drop(inference_subset.index, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset with {len(dataset)} records loaded!")
    return dataset


