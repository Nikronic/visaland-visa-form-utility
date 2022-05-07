__all__ = ['Logics', 'Discretizer', 'Merger']

from typing import List, Union
import pandas as pd


# TODO: implement issues #1 and #2 here
# TODO: extend issue #3 with aggregation methods experts suggest then implement them here
# check TODOs in preprocessor.py file too

class Logics:
    """
    Applies logics on different type of data resulting in summarized, expanded, or
        differently represented data for a specific purpose based on application.

    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        pass

    def occupation(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def field_of_study(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def count_items(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def count_items_living_in_foreign_country(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def has_items_living_in_foreign_country(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def count_items_born_in_foreign_country(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def has_items_born_in_foreign_country(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def count_highly_skilled_items(self, string: Union[List[str], str], target: str):
        raise NotImplementedError

    def has_highly_skilled_items(self, string: Union[List[str], str], target: str):
        raise NotImplementedError


class Discretizer:
    """


    """

    def __init__(self) -> None:
        pass

    def occupation(self, string: str, target: str) -> str:
        pass

    def field_of_study(self, string: str, target: str) -> str:
        pass


class Merger:
    """


    """

    def __init__(self) -> None:
        pass
