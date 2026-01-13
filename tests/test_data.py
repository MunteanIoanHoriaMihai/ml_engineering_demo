import pandas as pd
import pytest
from src.data_loader import load_data, clean_data

def test_load_data_returns_dataframe_and_series():
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]

def test_load_data_columns_and_target_name():
    X, y = load_data()
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    assert all(col in X.columns for col in expected_columns)
    assert y.name == "target"

def test_clean_data_no_missing_values():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    cleaned_df = clean_data(df)
    pd.testing.assert_frame_equal(cleaned_df, df)

def test_clean_data_with_missing_values():
    df_with_nan = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, 5, None]})
    cleaned_df = clean_data(df_with_nan)
    expected_df = pd.DataFrame({'col1': [1.0, 0.0, 3.0], 'col2': [4.0, 5.0, 0.0]})
    pd.testing.assert_frame_equal(cleaned_df, expected_df)

def test_clean_data_empty_dataframe():
    df_empty = pd.DataFrame()
    cleaned_df = clean_data(df_empty)
    pd.testing.assert_frame_equal(cleaned_df, df_empty)
