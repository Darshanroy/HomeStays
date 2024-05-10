from typing import Tuple
from zenml.logger import get_logger

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from steps.data_loader import data_loader
from zenml import step

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import (TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues,
                             TestNumberOfConstantColumns,
                             TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType,
                             TestNumberOfDriftedColumns)
from evidently.test_preset import NoTargetPerformanceTestPreset

ws = CloudWorkspace(
    token="dG9rbgHfd9TQMR1Mup4FnQPNmSJvQWvx7bmkvNLrLykS7wUcQwBQBxqDUNqQFT5kU07Oay7bwHNA5dH0To93wCHYuaGwFD09qdYPd1/BGYbdB/sMHR7K8veKFlPczNvT2Z6aPdkUog4Kb83dFezC9VqCaL1tk9pL29C6",
    url="https://app.evidently.cloud")
project = ws.get_project("0601979a-d2a7-4392-848f-62a6d9655df0")
logger = get_logger(__name__)


@step
def data_splitter(
        dataset: pd.DataFrame, test_size: float = 0.2
) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_trn"],
    Annotated[pd.DataFrame, "raw_dataset_tst"],
]:
    dataset_trn, dataset_tst = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    logger.info(f"Dataset Split completed!")

    dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)

    tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])

    tests.run(reference_data=dataset_trn, current_data=dataset_tst)
    ws.add_report(project.id, tests)

    suite = TestSuite(tests=[
        NoTargetPerformanceTestPreset(),
    ])

    suite.run(reference_data=dataset_trn, current_data=dataset_tst)
    ws.add_report(project.id, suite)

    return dataset_trn, dataset_tst
