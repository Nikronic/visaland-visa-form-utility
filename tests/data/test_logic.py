import pytest
import pandas as pd
from vizard.data.logic import CanadaLogics


class TestCanadaLogics:
    @pytest.fixture(autouse=True)
    def setup_instance(self):
        self.canada_logics = CanadaLogics()

    # --- average_age_group --- #

    def test_average_age_group_empty_series(self):
        series = pd.Series([])
        assert self.canada_logics.average_age_group(series) == 0.0

    def test_average_age_group_all_equal(self):
        series = pd.Series([25, 25, 25])
        assert self.canada_logics.average_age_group(series) == 25

    def test_average_age_group_mixed_ages(self):
        series = pd.Series([25, 30, 20, 15])
        expected_average = (25 + 30 + 20 + 15) / 4
        assert self.canada_logics.average_age_group(series) == expected_average

    # --- count_previous_residency_country --- #

    def test_count_previous_residency_country_empty_series(self):
        series = pd.Series([])
        assert self.canada_logics.count_previous_residency_country(series) == 0

    def test_count_previous_residency_country_all_non_zero(self):
        series = pd.Series([10, 5])
        assert self.canada_logics.count_previous_residency_country(series) == 2

    def test_count_previous_residency_country_mixed_zero_non_zero(self):
        series = pd.Series([10, 0])
        assert self.canada_logics.count_previous_residency_country(series) == 1

    # --- count_foreigner_family --- #

    def test_count_foreigner_family_empty_series(self):
        series = pd.Series([])
        assert self.canada_logics.count_foreigner_family(series) == 0

    def test_count_foreigner_family_all_iran(self):
        series = pd.Series(["iran", "iran", "iran"])
        assert self.canada_logics.count_foreigner_family(series) == 0

    def test_count_foreigner_family_all_foreign(self):
        series = pd.Series(["foreign", "foreign", "foreign"])
        assert self.canada_logics.count_foreigner_family(series) == len(series)

    def test_count_foreigner_family_mixed_iran_foreign(self):
        series = pd.Series(["iran", "foreign", "iran", "foreign"])
        assert self.canada_logics.count_foreigner_family(series) == 2

    # --- count_accompanying --- #

    def test_count_accompanying_empty_series(self):
        series = pd.Series([])
        assert self.canada_logics.count_accompanying(series) == 0

    def test_count_accompanying_all_true(self):
        series = pd.Series([True, True, True])
        assert self.canada_logics.count_accompanying(series) == len(series)

    def test_count_accompanying_all_false(self):
        series = pd.Series([False, False, False])
        assert self.canada_logics.count_accompanying(series) == 0

    def test_count_accompanying_mixed_true_false(self):
        series = pd.Series([True, False, False, True])
        assert self.canada_logics.count_accompanying(series) == 2

    # # TODO: why we counting None's as True too?
    # def test_count_accompanying_mixed_true_false_none(self):
    #     series = pd.Series([True, False, None, True])
    #     assert self.canada_logics.count_accompanying(series) == 2

    # --- count_rel --- #

    def test_count_rel_empty_series(self):
        series = pd.Series([])
        assert self.canada_logics.count_rel(series) == 0

    def test_count_rel_with_all_non_zero(self):
        series = pd.Series([1.3, 24, 31])
        assert self.canada_logics.count_rel(series) == 3

    def test_count_rel_with_some_non_zero(self):
        series = pd.Series([0, 24, 31])
        assert self.canada_logics.count_rel(series) == 2

    # --- count_long_distance_family_resident --- #

    def test_count_long_distance_family_resident_empty_series(self):
        series = pd.Series([], index=[])
        assert self.canada_logics.count_long_distance_family_resident(series) == 0

    def test_count_long_distance_family_resident_with_zero_long(self):
        series = pd.Series(
            ["alborz", "alborz", "alborz", "alborz"],
            index=["p1.SecA.App.AppAddr", 1, 2, 3],
        )
        assert self.canada_logics.count_long_distance_family_resident(series) == 0

    def test_count_long_distance_family_resident_with_long_none(self):
        series = pd.Series(
            ["alborz", "alborz", "alborz", None, None, None, "gilan", "isfahan", None],
            index=["p1.SecA.App.AppAddr", 1, 2, 3, 4, 5, 6, 7, 8],
        )
        assert self.canada_logics.count_long_distance_family_resident(series) == 2

    # --- count_foreign_family_resident --- #

    def test_count_foreign_family_resident_empty_series(self):
        series = pd.Series([], index=[])
        assert self.canada_logics.count_foreign_family_resident(series) == 0

    def test_count_foreign_family_resident_with_zero_foreign_with_none(self):
        series = pd.Series([[None, None, "alborz", "fars"]])
        assert self.canada_logics.count_foreign_family_resident(series) == 0

    def test_count_foreign_family_resident_with_foreign_without_none(self):
        series = pd.Series(["foreign", "foreign", "alborz", "fars"])
        assert self.canada_logics.count_foreign_family_resident(series) == 2

    def test_count_foreign_family_resident_with_foreign_none(self):
        series = pd.Series(
            ["alborz", "alborz", None, "foreign", None, "gilan", "isfahan"]
        )
        assert self.canada_logics.count_foreign_family_resident(series) == 1
