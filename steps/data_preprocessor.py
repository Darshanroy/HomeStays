from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pickle
from steps.data_loader import data_loader
from steps.data_splitter import data_splitter
logging = get_logger(__name__)
import os

import pandas as pd
from sqlalchemy import create_engine

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset

ws = CloudWorkspace(token="dG9rbgHfd9TQMR1Mup4FnQPNmSJvQWvx7bmkvNLrLykS7wUcQwBQBxqDUNqQFT5kU07Oay7bwHNA5dH0To93wCHYuaGwFD09qdYPd1/BGYbdB/sMHR7K8veKFlPczNvT2Z6aPdkUog4Kb83dFezC9VqCaL1tk9pL29C6", url="https://app.evidently.cloud")

project = ws.get_project("0601979a-d2a7-4392-848f-62a6d9655df0")

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


def count_amenities(amenities_str):
    """
    Function to count the number of amenities
    """
    amenities = amenities_str.strip('{}').replace('"', '').split(',')
    return len(amenities)


def clean_zipcode(zipcode):
    """
    Clean the zipcode column by converting to integer if length is 5.
    """
    if pd.isnull(zipcode):
        return None
    zipcode_str = str(zipcode)
    if len(zipcode_str) == 5:
        return int(zipcode_str)
    else:
        return None


def label_encode_column(df, column_name):
    """
    Perform label encoding on a column in a DataFrame.
    """
    label_encoder = LabelEncoder()
    df[column_name + '_encoded'] = label_encoder.fit_transform(df[column_name])
    return df, label_encoder


def random_imputation(df, column):
    """
    Perform random imputation for missing values in a specific column of a DataFrame.
    """
    col = df[column]
    random_values = np.random.choice(col.dropna(), size=col.isnull().sum(), replace=True)
    col[col.isnull()] = random_values
    df[column] = col
    return df


def knn_impute_missing_values(df, columns_with_missing, n_neighbors=5):
    """
    Impute missing values in specified columns using KNN imputation.
    """
    for column in columns_with_missing:
        df_missing = df[[column]]
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df_missing)
        imputed_df = pd.DataFrame(imputed_data, columns=[column])
        imputed_df[column] = imputed_df[column].round().astype(int)
        df[column] = imputed_df[column]
    return df


def label_encode_categorical_columns(df, categorical_columns):
    """
    Apply label encoding to specified categorical columns in the DataFrame.
    """
    label_encoders = {}
    encoded_df = df.copy()
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        encoded_df[column] = label_encoder.fit_transform(encoded_df[column])
        label_encoders[column] = label_encoder
    return encoded_df, label_encoders


def extract_amenities(amenities_str):
    """
    Extract amenities from a JSON string and create a list of all amenities.
    """
    amenities = amenities_str.strip('{}').replace('"', '').split(',')
    return amenities


def check_important_amenity(amenities_str, important_amenity):
    """
    Checks if an important amenity is present in the amenities string.
    """
    return 1 if important_amenity in amenities_str else 0

@step(enable_cache=True)
def data_preprocessor(
        df: pd.DataFrame,
        normalize: Optional[bool] = None,
        drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:

    # Clean the 'zipcode' column
    logging.info("Cleaning the 'zipcode' column")
    df['zipcode'] = df['zipcode'].apply(clean_zipcode)

    # Label encode the 'neighbourhood' column and handle missing values
    logging.info("Label encoding the 'neighbourhood' column")
    df, label_encoder = label_encode_column(df, 'neighbourhood')
    null_indices = df[df['neighbourhood'].isnull()].index.tolist()
    df.loc[null_indices, 'neighbourhood_encoded'] = float('nan')

    # Specify the columns with missing values
    columns_with_missing = ['bathrooms', 'review_scores_rating', 'bedrooms', 'beds', 'zipcode', 'neighbourhood_encoded']

    # Impute missing values using KNN imputation
    logging.info("Imputing missing values using KNN imputation")
    df = knn_impute_missing_values(df, columns_with_missing)

    # Clean and convert 'host_response_rate' column to integer
    logging.info("Cleaning and converting 'host_response_rate' column to integer")
    df['host_response_rate'] = df['host_response_rate'].str.split("%").str[0]
    df = random_imputation(df, column='host_response_rate')
    df['host_response_rate'] = df['host_response_rate'].astype('int')

    # Convert multiple columns to datetime format
    logging.info("Converting multiple columns to datetime format")
    df[['last_review', 'host_since', 'first_review']] = df[['last_review', 'host_since', 'first_review']].apply(
        pd.to_datetime)

    # Forward fill missing datetime values
    logging.info("Forward filling missing datetime values")
    df['last_review'] = df['last_review'].fillna(method='ffill')
    df['host_since'] = df['host_since'].fillna(method='ffill')
    df['first_review'] = df['first_review'].fillna(method='ffill')



    # Impute missing values for 'host_identity_verified' and 'host_has_profile_pic' columns
    logging.info("Imputing missing values for 'host_identity_verified' and 'host_has_profile_pic' columns")
    df = random_imputation(df, column='host_identity_verified')
    df = random_imputation(df, column='host_has_profile_pic')

    # Feature engineering for review duration and time since last review
    logging.info("Feature engineering for review duration and time since last review")
    df['review_duration'] = (df['last_review'] - df['first_review']).dt.days
    df['time_since_last_review'] = (pd.to_datetime('today') - df['last_review']).dt.days

    # Feature engineering for host tenure
    logging.info("Feature engineering for host tenure")
    df['host_tenure'] = (pd.to_datetime('today') - df['host_since']).dt.days

    # Calculate the average review score for each 'number_of_reviews' group
    logging.info("Calculating the average review score for each 'number_of_reviews' group")
    average_review_score = df.groupby('number_of_reviews')['review_scores_rating'].mean().reset_index()
    average_review_score.rename(columns={'review_scores_rating': 'average_review_score'}, inplace=True)

    # Merge the average_review_score back to the original DataFrame
    logging.info("Merging the average_review_score back to the original DataFrame")
    df = pd.merge(df, average_review_score, on='number_of_reviews', how='left')

    # Extract all amenities and create a list of all amenities across all listings
    logging.info("Extracting all amenities and creating a list of all amenities across all listings")
    all_amenities = []
    for amenities_str in df['amenities']:
        all_amenities.extend(extract_amenities(amenities_str))

    # Calculate the frequency of each amenity
    logging.info("Calculating the frequency of each amenity")
    amenities_frequency = pd.Series(all_amenities).value_counts()

    # Select the top 5 most common amenities
    logging.info("Selecting the top 5 most common amenities")
    top_5_amenities = amenities_frequency.head(5).index.tolist()


    # copying the list of important amenities
    logging.info("Copying the list of important amenities")
    important_amenities = top_5_amenities

    # Create binary columns for each important amenity with a prefix
    prefix = 'has_'
    logging.info("Creating binary columns for each important amenity with a prefix")
    for amenity in important_amenities:
        column_name = prefix + amenity.lower().replace(' ', '_')
        df[column_name] = df['amenities'].apply(lambda x: check_important_amenity(x, amenity))

    # Extract month from datetime columns
    df['last_review_month'] = df['last_review'].dt.month
    df['host_since_month'] = df['host_since'].dt.month
    df['first_review_month'] = df['first_review'].dt.month

    # Drop original datetime columns
    df.drop(columns=['last_review', 'host_since', 'first_review'], inplace=True)

    # Drop unnecessary columns
    logging.info("Dropping unnecessary columns")
    df = df.drop(["thumbnail_url", "neighbourhood", "amenities", 'description'], axis=1)

    # List of categorical columns to be label encoded
    logging.info("Listing categorical columns to be label encoded")
    categorical_columns = ['property_type', 'room_type', 'bed_type',
                           'cancellation_policy', 'city', 'name',
                           'host_has_profile_pic', 'host_identity_verified',
                           'instant_bookable', 'cleaning_fee']

    logging.info("Encoding categorical columns")
    encoded_df, label_encoders = label_encode_categorical_columns(df, categorical_columns)

    data_report = Report(
        metrics=[
            DataQualityPreset(),
        ],
    )
    data_report.run(reference_data=None, current_data=encoded_df)
    ws.add_report(project.id, data_report)

    data_report.save_html("templates/file.html")
    push_to_sql(encoded_df.copy(), engine, table_name)

    return encoded_df

