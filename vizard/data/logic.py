__all__ = [
    'Logics', 'CanadaLogics'
]

# core
from functools import reduce
import pandas as pd
import numpy as np
# helpers
from typing import Callable, cast


class Logics:
    """Applies logics on different type of data resulting in summarized, expanded, or transformed data

    Methods here are implemented in the way that can be used as Pandas.agg_ function
    over `Pandas.Series` using functools.reduce_.

    Note: 
        This is constructed based on domain knowledge hence is designed 
        for a specific purpose based on application.

    .. _Pandas.agg: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.agg.html
    .. _functools.reduce: https://docs.python.org/3/library/functools.html#functools.reduce
    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        """Init class by setting dataframe globally

        Args:
            dataframe (pd.DataFrame, optional): The dataframe that functions of this class 
                will be user over its series, i.e. ``Logics.*(series)``. Defaults to None.
        """
        self.df = dataframe

    def __check_df(self, func: str) -> None:
        """Checks that `self.df` is initialized when function with the name ``func`` is being called

        Args:
            func (str): The name of the function that operates over ``self.df``

        Raises:
            TypeError: If ``self.df`` is not initialized
        """
        if self.df is None:
            raise TypeError(
                f'`df` attribute cannot be `None` when using "{func}".')

    def reset_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Takes a new dataframe and replaces the old one

        Note:
            This should be used when the dataframe is modified outside of functions 
            provided in this class. E.g.::

                my_df: pd.DataFrame = ...
                logics = Logics(dataframe=my_df)
                my_df = third_party_tools(my_df)
                # now update df in logics
                logics.reset_dataframe(dataframe=my_df)

        Args:
            dataframe (pd.DataFrame): The new dataframe
        """
        self.df = dataframe

    def add_agg_column(
        self,
        aggregator: Callable,
        agg_column_name: str,
        columns: list
    ) -> pd.DataFrame:
        """Aggregate columns and adds it to the original dataframe using an aggregator function

        Args:
            aggregator (Callable): A function that takes multiple columns of a
                series and reduces it
            agg_column_name (str): The name of new aggregated column
            columns (list): Name of columns to be aggregated (i.e. input to ``aggregator``)

        Note:
            Although this function updated the dataframe the class initialized with *inplace*,
            but user must update the main dataframe outside of this class to make sure he/she
            can use it via different tools. Simply put::

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
    
    def average_age_group(self, series: pd.Series) -> float:
        """Calculates the average age of a specific group

        Args:
            series (pd.Series): Pandas Series to be processed

        Returns:
            float: Result of averaging
        """
        raise NotImplementedError

    def count_previous_residency_country(self, series: pd.Series) -> int:
        """Counts the number of previous country of resident

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_foreigner_family(self, series: pd.Series) -> int:
        """Counts the number of family members born in a foreign country

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_accompanying(self, series: pd.Series) -> int:
        """Counts the number of people that are accompanying main person

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_rel(self, series: pd.Series) -> int:
        """Counts the number of items for the given relationship

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed

        Returns:
            int: Result of counting
        """
        raise NotImplementedError

    def count_long_distance_family_resident(self, series: pd.Series) -> int:
        """Counts the number of family members that are long distance resident

        Note:
            Those who are living in another country may not be considered as
            long distance resident. In that case, 
            see :func:`count_foreign_family_resident` for more info.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed

        Returns:
            int: Result of counting
        """

        raise NotImplementedError

    def count_foreign_family_resident(self, series: pd.Series) -> int:
        """Counts the number of family members that are living in a foreign country

        Note:
            This is an special case of :func:`count_long_distance_family_resident` 
            where those who are living in another country are treated
            separately. In case you don't care, just use 
            :func:`count_long_distance_family_resident` instead.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed

        Returns:
            int: Result of counting
        """

        raise NotImplementedError


class CanadaLogics(Logics):
    """
    Customize and extend logics defined in :class:`Logics` to Canada dataset.

    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        super().__init__(dataframe)

    def average_age_group(self, series: pd.Series) -> float:
        """Calculates the average age of a specific group (e.g., children)

        This is used when we have many instances in a group that each instance
        by themselves don't provide much info; particularly when they are
        highly correlated with a main factor such as the age of the applicant 
        him/herself. In this case, we rather have the aggregated mode (average).

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed
                containing age (floats)

        Returns:
            float: Result of averaging
        """

        def sum(x, y): return np.sum([x, y])
        return reduce(lambda x, y: sum(x, y), series, 0.) / len(series)

    def count_previous_residency_country(self, series: pd.Series) -> int:
        """Counts the number of previous residency by counting non-zero periods of residency

        When ``*.Period == 0``, then we can say that the person has no residency.
        This way one just needs to count non-zero periods.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed containing
                residency periods

        Returns:
            int: Result of counting
        """

        def counter(x, y): return np.sum(np.isin([x, y], [0]))
        return reduce(lambda x, y: 2 - counter(x, y), series)

    def count_foreigner_family(self, series: pd.Series) -> int:
        """Counts the number of family members born in foreign country

        This has been hardcoded here by checking non-``'iran'``
        country of birth.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed containing 
                country of birth of members

        Returns:
            int: Result of counting
        """

        def counter(y): return np.sum(
            np.invert(np.isin([y], ['iran', None])))  # type: ignore
        return reduce(lambda x, y: x + counter(y), series, 0)

    def count_accompanying(self, series: pd.Series) -> int:
        """Counts the number of people that are accompanying the main person

        This has been done by checking the corresponding bool flag

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed containing
                accompany binary/bool status

        Returns:
            int: Result of counting
        """

        def counter(y): return np.sum(
            np.isin([y], [True, None]))  # type: ignore
        # type: ignore
        return reduce(lambda x, y: x + counter(y), series, False)

    def count_rel(self, series: pd.Series) -> int:
        """Counts the number of people for the given relationship, e.g. siblings.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed 

        Returns:
            int: Result of counting
        """

        def counter(y): return np.sum(y != 0.)
        return reduce(lambda x, y: x + counter(y), series, 0)

    def count_long_distance_family_resident(self, series: pd.Series) -> int:
        """Counts the number of family members that are long distance resident

        This is being done comparing applicants' province with their families' province.
        This will ignore ``'deceased'`` too.

        Note:
            Those who are living in another country (in our dataset as
            "foreign") are not considered as long distance resident. In
            that case, see :func:`count_foreign_family_resident` for more info.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed containing 
                the residency state/province in string. In practice, 
                any string different from applicant's province will be counted
                as difference.

        Examples:
            >>> import pandas as pd
            >>> from vizard.data.logic import CanadaLogics
            >>> f = CanadaLogics().count_long_distance_family_resident
            >>> s = pd.Series(['alborz', 'alborz', 'alborz', None, None, None, 'gilan', 'isfahan', None],
            >>>    index=['p1.SecA.App.AppAddr', 1, 2, 3, 4, 5, 6, 7, 8])
            >>> f(s)
            2
            >>> s1 = pd.Series(['alborz', 'alborz', 'alborz', 'alborz'], 
            >>>    index=['p1.SecA.App.AppAddr', '1', '2', '3'])
            >>> f(s1)
            0


        Returns:
            int: Result of counting
        """

        self.df = cast(pd.DataFrame, self.df)  # for mypy only

        apps_loc: str = series['p1.SecA.App.AppAddr']

        def counter(y):
            return np.sum(
                np.invert(
                    np.isin([y], [apps_loc, None, 'foreign', 'deceased']),  # type: ignore
                    dtype=bool
                )
            )
        return reduce(lambda x, y: x + counter(y), series, 0)

    def count_foreign_family_resident(self, series: pd.Series) -> int:
        """Counts the number of family members that are long distance resident

        This is being done by only checking the literal value ``'foreign'`` in the
        ``'*Addr'`` columns (address columns).

        Note:
            Those who are living in another city are not considered as "foreign". In
            that case, see :func:`count_long_distance_family_resident` for more info.

        Args:
            series (:class:`pandas.Series`): Pandas Series to be processed containing 
                the residency state/province in string. In practice, 
                any string different from applicant's province will be counted
                as difference.

        Examples:
            >>> import pandas as pd
            >>> from vizard.data.logic import CanadaLogics
            >>> f = CanadaLogics().count_foreign_family_resident
            >>> s = pd.Series(['alborz', 'alborz', 'alborz', None, 'foreign', None, 'gilan', 'isfahan', None])
            >>> f(s)
            1
            >>> s1 = pd.Series(['foreign', 'foreign', 'alborz', 'fars'])
            >>> f(s1)
            2
            >>> s2 = pd.Series([None, None, 'alborz', 'fars'])
            >>> f(s2)
            0

        Returns:
            int: Result of counting
        """

        self.df = cast(pd.DataFrame, self.df)  # for mypy only

        def counter(y): return np.sum(
            np.isin([y], ['foreign']))  # type: ignore
        return reduce(lambda x, y: x + counter(y), series, 0)  # type: ignore
