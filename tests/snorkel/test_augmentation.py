import pandas as pd
import pytest
from typing import cast

from vizard.snorkel.augmentation import SeriesNoise


class TestSeriesNoise:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        data = [
            ["A", 1.1, 1],
            ["B", 1.2, 2],
            ["C", 1.3, 3],
        ]

        df = pd.DataFrame(data=data, columns=["categorical", "real", "ordered"])
        return df

    def test_init_without_dataframe(self):
        sn = SeriesNoise()
        assert sn.df is None

    def test_init_with_dataframe(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        sn = SeriesNoise(df)
        assert sn.df.equals(df)

    def test_set_dataframe(self):
        sn = SeriesNoise()
        df = pd.DataFrame({"A": [1, 2, 3]})
        sn.set_dataframe(df)
        assert sn.df.equals(df)

    def test_series_add_normal_noise_with_dataframe(self, sample_df):
        sample_df = cast(pd.DataFrame, sample_df)
        sn = SeriesNoise(sample_df)
        s_noisy = sn.series_add_normal_noise(sample_df.iloc[0].copy(), "real")
        assert not sample_df.iloc[0].equals(s_noisy)

    def test_series_add_truncated_normal_noise_with_dataframe(self, sample_df):
        sn = SeriesNoise(sample_df)
        s_noisy = sn.series_add_truncated_normal_noise(
            sample_df.iloc[0].copy(), "real", mean=0, std=1, lb=-0.9, ub=0.9
        )
        assert not sample_df.iloc[0].equals(s_noisy)

    def test_series_add_truncated_normal_noise_with_dataframe(self, sample_df):
        sn = SeriesNoise(sample_df)
        s_noisy = sn.series_add_truncated_normal_noise(
            sample_df.iloc[0].copy(), "real", mean=0, std=1, lb=-0.9, ub=0.9
        )
        assert not sample_df.iloc[0].equals(s_noisy)

    def test_categorical_switch_noise_with_dataframe(self, sample_df):
        sn = SeriesNoise(sample_df)
        categories_map = {"A": "B", "B": "C"}
        s_noisy = sn.categorical_switch_noise(
            s=sample_df.iloc[0].copy(), column="categorical", categories=categories_map
        )
        assert not sample_df.iloc[0].equals(s_noisy)

    def test_series_add_ordered_noise_with_dataframe(self, sample_df):
        sample_df = cast(pd.DataFrame, sample_df)
        sn = SeriesNoise(sample_df)
        s_noisy = sn.series_add_ordered_noise(
            s=sample_df.iloc[0].copy(), column="ordered", lb=1, ub=4
        )
        assert not sample_df.iloc[0].equals(s_noisy)

    def test_series_add_ordered_noise_single_choice(self, sample_df):
        sample_df = cast(pd.DataFrame, sample_df)
        sn = SeriesNoise(sample_df)
        s_noisy = sn.series_add_ordered_noise(
            s=sample_df.iloc[0].copy(), column="ordered", lb=1, ub=3
        )
        assert s_noisy["ordered"] == 2
        s_noisy = sn.series_add_ordered_noise(
            s=sample_df.iloc[2].copy(), column="ordered", lb=2, ub=4
        )
        assert s_noisy["ordered"] == 2
