import pandas as pd
import pytest

from vizard.data.functional import column_dropper


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2_to_drop": [4, 5, 6],
            "col3_keep": [7, 8, 9],
            "col4_regex_match": [10, 11, 12],
        }
    )


class TestColumnDropper:
    def test_column_dropper_basic(self, sample_df):
        result = column_dropper(
            dataframe=sample_df, string="col2_to_drop", inplace=True, regex=False
        )
        assert result is None
        assert "col2_to_drop" not in sample_df
        assert set(sample_df.columns) == {"col1", "col3_keep", "col4_regex_match"}

    def test_column_dropper_exclude(self, sample_df):
        result = column_dropper(
            dataframe=sample_df,
            regex=True,
            string="col.*",
            exclude="keep",
            inplace=True,
        )
        assert result is None
        assert set(sample_df.columns) == {"col3_keep"}

    def test_column_dropper_not_inplace(self, sample_df):
        result = column_dropper(
            dataframe=sample_df, string="col2_to_drop", inplace=False, regex=False
        )
        assert isinstance(result, pd.DataFrame)
        assert "col2_to_drop" not in result.columns
        assert "col2_to_drop" in sample_df.columns  # Original DataFrame is unmodified
