__all__ = [
    'Logics', 'CanadaLogics'
]

# core
from functools import reduce
import pandas as pd
import numpy as np
# helpers
from typing import Callable, List, Union, cast


# TODO: extend issue #3 with aggregation methods experts suggest then implement them here
# check TODOs in preprocessor.py file too

class Logics:
    """Applies logics on different type of data resulting in summarized, expanded, or
        differently represented data
    
    Methods here are implemented in the way that can be used as `agg` [#]_ function
        over `Pandas.Series` using `functools.reduce` [#]_.
    
    Note: 
        This is constructed based on domain knowledge hence is designed 
            for a specific purpose based on application.
    
    .. [#] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.agg.html
    .. [#] https://docs.python.org/3/library/functools.html#functools.reduce
    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        """Init class by setting dataframe globally

        Args:
            dataframe (pd.DataFrame, optional): The dataframe that functions of this class 
                will be user over its series, i.e. ``Logics.*(series)``. Defaults to None.
        """
        self.df = dataframe

    def __check_df(self, func: str) -> None:
        """Checks that `self.df` is initialized when function with the name `func` is being called

        Args:
            func (str): The name of the function that operates over `self.df`

        Raises:
            TypeError: If `self.df` is not initialized
        """
        if self.df is None:
            raise TypeError(
                f'`df` attribute cannot be `None` when using "{func}".')

    def reset_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Takes a new dataframe and replaces the old one

        Note:
            This should be used when the dataframe is modified outside of functions 
                provided in this class. E.g.
                ::
                    my_df: pd.DataFrame = ...
                    logics = Logics(dataframe=my_df)
                    my_df = third_party_tools(my_df)
                    # now update df in logics
                    logics.reset_dataframe(dataframe=my_df)

        Args:
            dataframe (pd.DataFrame): The new dataframe
        """
        self.df = dataframe

    def add_agg_column(self, aggregator: Callable,
                       agg_column_name: str,
                       columns: list) -> pd.DataFrame:
        """Aggregate columns and adds it to the original dataframe using an aggregator function

        Args:
            aggregator (Callable): A function that takes multiple columns of a
                series and reduces it
            agg_column_name (str): The name of new aggregated column
            columns (list): Name of columns to be aggregated (i.e. input to `aggregator`)

        Note:
            Although this function updated the dataframe the class initialized with *inplace*,
                but user must update the main dataframe outside of this class to make sure he/she
                can use it via different tools. Simply put
                ::
                    my_df: pd.DataFrame = ...
                    logics = Logics(dataframe=my_df)
                    my_df = logics.add_agg_column(...)
                    my_df = third_party_tools(my_df)
                    # now update df in logics
                    logics.reset_dataframe(dataframe=my_df)
                    # aggregate again...
                    my_df = logics.add_agg_column(...)

        Returns:
            pd.DataFrame: Updated dataframe that contains aggregated data
        """
        # check self.df is initialized
        self.__check_df(func=self.add_agg_column.__name__)
        self.df = cast(pd.DataFrame, self.df)
        # aggregate
        self.df[agg_column_name] = self.df[columns].agg(aggregator, axis=1)
        self.df = self.df.rename(
            columns={aggregator.__name__: agg_column_name})
        # return updated dataframe to be used outside of this class
        return self.df

    def count_previous_residency_country(self, series: pd.Series) -> int:
        """Counts the number of previous country of resident

        Args:
            series (pd.Series): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_foreigner_family(self, series: pd.Series) -> int:
        """Counts the number of family members born in a foreign country

        Args:
            series (pd.Series): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_accompanying(self, series: pd.Series) -> int:
        """Counts the number of people that are accompanying main person

        Args:
            series (pd.Series): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_rel(self, series: pd.Series) -> int:
        """Counts the number of items for the given relationship

        Args:
            series (pd.Series): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError


class CanadaLogics(Logics):
    """
    Customize and extend logics defined in `logic.Logics` to Canada dataset.

    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        super().__init__(dataframe)

    def count_previous_residency_country(self, series: pd.Series) -> int:
        """Counts the number of previous residency by counting non-zero periods of residency

        When ``*.Period == 0``, then we can say that the person has no residency.
            This way one just needs to count non-zero periods.

        Args:
            series (pd.Series): Pandas Series to be processed containing
                residency periods

        Returns:
            int: Result of counting
        """

        counter = lambda x, y: np.sum(np.isin([x, y], [0]))
        return reduce(lambda x, y: 2 - counter(x, y), series)

    def count_foreigner_family(self, series: pd.Series) -> int:
        """Counts the number of family members born in foreign country
        
        This has been hardcoded here by checking non-``'iran'``
            country of birth.

        Args:
            series (pd.Series): Pandas Series to be processed containing 
                country of birth of members

        Returns:
            int: Result of counting
        """

        counter = lambda y: np.sum(np.invert(np.isin([y], ['iran', None])))  # type: ignore
        return reduce(lambda x, y: x + counter(y), series, 0)

    def count_accompanying(self, series: pd.Series) -> int:
        """Counts the number of people that are accompanying the main person
        
        This has been done by checking the corresponding bool flag

        Args:
            series (pd.Series): Pandas Series to be processed containing
                accompany binary/bool status

        Returns:
            int: Result of counting
        """

        counter = lambda y: np.sum(np.isin([y], [True, None]))  # type: ignore
        return reduce(lambda x, y: x + counter(y), series, False)  # type: ignore

    def count_rel(self, series: pd.Series) -> int:
        """Counts the number of people for the given relationship,
            e.g. number of children, siblings, etc.

        Args:
            series (pd.Series): Pandas Series to be processed 

        Returns:
            int: Result of counting
        """

        counter = lambda y: np.sum(y != 0.)
        return reduce(lambda x, y: x + counter(y), series, 0)
