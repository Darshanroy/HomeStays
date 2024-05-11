from typing import Tuple
from zenml.logger import get_logger
from sqlalchemy import create_engine

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from steps.data_loader import data_loader
from zenml import step
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import DataDriftPreset
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


def select_features_with_mi_and_log_price(data: pd.DataFrame, target_column: str, num_features: int = 20) -> pd.DataFrame:
    """
    Performs feature selection using mutual information regression and returns a DataFrame
    containing the selected features along with the target column (log_price).

    Args:
        data (pd.DataFrame): The input DataFrame containing features.
        target_column (str): The name of the target column (assumed to be 'log_price').
        num_features (int, optional): The number of features to select. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing the selected features and the target column.
    """

    if target_column != 'log_price':
        raise ValueError("target_column must be 'log_price' to include it in the output")

    X = data.drop(target_column, axis=1)  # Extract features
    y = data[target_column]  # Extract target variable (assuming it's 'log_price')

    k_best = SelectKBest(mutual_info_regression, k=num_features)
    X_selected = k_best.fit_transform(X, y)  # Feature selection

    selected_features = X.columns[k_best.get_support(indices=True)]

    # Create a DataFrame containing selected features and the target column
    selected_df = pd.concat([X[selected_features], data[[target_column]]], axis=1)

    return selected_df


# Replace with your database connection details
engine = create_engine("sqlite://", echo=True)

# Table name
table_name = 'Encoded_Data'

def push_to_sql(df, engine, table_name):
  """Pushes a DataFrame to a SQL table using SQLAlchemy.

  Args:
      df (pandas.DataFrame): The DataFrame to push.
      engine (sqlalchemy.engine.Engine): The SQLAlchemy engine object.
      table_name (str): The name of the table to create or insert into.
  """

  try:
    # Check if table exists. If not, create it with appropriate data types.
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Data successfully pushed to table '{table_name}'.")
  except Exception as e:
    print(f"Error pushing data to table: {e}")




@step
def data_splitter(
        dataset: pd.DataFrame, test_size: float = 0.2
) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_trn"],
    Annotated[pd.DataFrame, "raw_dataset_tst"],
]:
    selected_features_df = select_features_with_mi_and_log_price(dataset.copy(), target_column='log_price', num_features=10)
    logger.info(f"Feature Selection completed!")

    dataset_trn, dataset_tst = train_test_split(
        selected_features_df,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    logger.info(f"Dataset Split completed!")

    logger.info(f"Selected Columns : {list(selected_features_df.columns)}")

    dataset_trn = pd.DataFrame(dataset_trn, columns=selected_features_df.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=selected_features_df.columns)

    data_report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset()
        ],
    )
    data_report.run(reference_data=dataset_trn, current_data=dataset_tst)
    ws.add_report(project.id, data_report)

    data_report.save_html("templates/file.html")


    push_to_sql(dataset_trn.copy(), engine, table_name)












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
    suite.save_html("templates/test_suite.html")

    return dataset_trn, dataset_tst
