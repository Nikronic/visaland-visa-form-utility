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

    def standardize_occupation(self, string: Union[List[str], str], target: str):
        """
        Takes a list of occupations in string and converts them into predetermined
            categories, e.g. nurse-> medical, engineer


        """
        raise NotImplementedError

    def is_highly_skilled(self, string: str) -> bool:
        """
        Takes a job category (from `standardize_occupation`) or job title and tells if it's
            a highly skilled job or not.

        """
        raise NotImplemented

    def determine_skill_level(self, string: str) -> int:
        """
        Takes a job category (from `standardize_occupation`) or job title and tells 
            how high skill the job is. (0=lowest, 1=mediocre, 2=highest)

        """

        raise NotImplementedError

    def standardize_field_of_study(self, string: Union[List[str], str], target: str):
        """
        Takes a list of study fields in string and converts them into predetermined
            categories, e.g. technician, specialist, manager, etc. (see job hirerachy)

        """
        raise NotImplementedError

    def count_items(self, string: Union[List[str], str], target: str):
        """
        Counts items of the same type in a list of strings.\n
        Note that the input should be already preprocessed using other
            methods available first.

        """
        raise NotImplementedError

    def has_items(self, string: Union[List[str], str], target: str):
        """
        Checks whether or not a list of strings contains a specific item.\n 
        Note that the input should be preprocessed using other methods available first.

        """
        raise NotImplementedError


class Discretizer:
    """


    """

    def __init__(self) -> None:
        pass


class Merger(Logics):
    """
    Combines multiple features into one by applying the "logic" functions

    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        super().__init__(dataframe)

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


