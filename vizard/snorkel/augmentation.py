# for forward reference https://stackoverflow.com/a/50768146/18971263
from __future__ import annotations

__all__ = [
    'SeriesNoise'
]

# core
from scipy.stats import truncnorm
import pandas as pd
import numpy as np
# snorkel
from snorkel.augmentation import transformation_function
from snorkel.augmentation import TransformationFunction
# helpers
from typing import Any, Callable, Optional, Tuple, Union, cast
from enum import Enum

class SeriesNoise:
    """
    Adds different type noise (multiplicative or additive) to a `Pandas.Series`

    Examples:
        >>> from vizard.snorkel import augmentation
        >>> series_noise_augmenter = augmentation.SeriesNoise(dataframe=data)
        >>> tf_Funds = series_noise_augmenter.make_tf(func=series_noise_augmenter.series_add_normal_noise, 
                            column='P3.DOV.PrpsRow1.Funds.Funds', mean=0, std=1000.)
        >>> tf_DOBYear = series_noise_augmenter.make_tf(func=series_noise_augmenter.series_add_normal_noise, 
                            column='P1.PD.DOBYear.Period', mean=0, std=5.)
    """

    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        """
        
        args:
            dataframe: The dataframe that series will be extracted from to be processed
        """
        self.rng = np.random.default_rng()  # rng for all random generation
        self.df = dataframe  # the main dataframe that series are extracted

    def __check_dataframe_initialized(self) -> None:
        if self.df is None:
            raise ValueError('You cannot use `series_*` functions before initializing main\
                             main dataframe (`self.df`). Call `set_dataframe(df)` first.')

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        To initialize main dataframe if not initialized already at the time of 
            instance creation, i.e. `... = SeriesNoise(...)`
        
        **Must be called before calling any of `series_*` functions.**
        """
        self.df = df

    def make_tf(self, func: Callable, **kwargs) -> TransformationFunction:
        """Make any function an instance of `TransformationFunction`

        Note:
            Currently only `func`s that work on `pd.Series` that work on single `column`
                are supported as the API is designed this way. But, it could be easily
                modified to support any sort of function.

        Args:
            func (Callable): A callable that
            column (str): column name of a `pd.Series` that is going to be manipulated.
                Must be provided if `func` is not setting it internally. E.g. for
                `CanadaAugmentation.tf_add_normal_noise_dob_year` you don't need to 
                set `column` since it is being handled internally.
            kwargs: keyword arguments to `func`

        Returns:
            TransformationFunction: A callable object that is now compatible with
                `snorkel` transformation pipeline, e.g. ``Policy`` and ``TFApplier``
        """

        if 'column' in kwargs:
            column = kwargs.pop('column')
            return TransformationFunction(
                name=f'{func.__name__}_{column}',
                f=func, resources=dict(column=column, **kwargs),
        )    
        else:
            return TransformationFunction(
                name=f'{func.__name__}',
                f=func, resources=dict(**kwargs),
            )

    def normal_noise(self, mean: float, std: float,
                     size: Union[Tuple[int, ...], int]) -> np.ndarray:
        """A wrapper around `numpy.Generator.normal`. 

        Dev may extend this method by adding features to it; e.g. adding it to 
            a pandas Series (see `self.add_normal_noise`)

        """
        return self.rng.normal(loc=mean, scale=std, size=size)

    def series_add_normal_noise(self, s: pd.Series, column: str,
                                mean: float = 0. , std: float = 1.) -> pd.Series:
        """Takes a pandas Series and corresponding column and adds *normal* noise to it

        Args:
            s (pd.Series): Pandas Series to be manipulated (from `self.df`)
            column (str): corresponding column in `self.df` and `s` 
            mean (float, optional): mean of normal noise. Defaults to 0.
            std (float, optional): standard deviation of normal noise. Defaults to 1.

        Returns:
            pd.Series: Noisy `column` of `s`
        """
        self.__check_dataframe_initialized()
        self.df = cast(pd.DataFrame, self.df)
        assert np.isscalar(s[column])

        # add noise
        noise: float = self.normal_noise(
            mean=mean, std=std, size=1).item()  # must be ndim == 0
        s[column] = s[column] + noise
        return s

    def truncated_normal_noise(self, mean: float, std: float,
                               low: float, upp: float,
                               size: Union[Tuple[int, ...], int] = 1) -> np.ndarray:
        """A wrapper around `scipy.stats.truncnorm` with lower/upper bound

        Note:
            Dev may extend this method by adding features to it; e.g. adding it to 
                a pandas Series (see `series_add_truncated_normal_noise`)

        Reference:
            1. https://stackoverflow.com/a/44308018/18971263

        Args:
            mean (float): mean of normal distribution
            std (float): standard deviation of normal distribution
            low (float): lower bound for truncation
            upp (float): upper bound for truncation
            size (Union[Tuple[int, ...], int], optional): shape of samples

        Returns:
            np.ndarray: A truncated normal distribution
        """
        return truncnorm((low - mean) / std, (upp - mean) / std,
                         loc=mean, scale=std).rvs(size)

    def series_add_truncated_normal_noise(self, s: pd.Series, column: str,
                                          mean: float, std: float, 
                                          lb: float, ub: float) -> pd.Series:
        """Takes a pandas Series and corresponding column and adds *truncated normal* noise to it

        Args:
            s (pd.Series): Pandas Series to be manipulated (from `self.df`)
            column (str): corresponding column in `self.df` and `s` 
            mean (float):  mean of normal noise
            std (float): standard deviation of normal noise
            lb (float): lower bound for truncation
            ub (float): higher bound for truncation

        Returns:
            pd.Series: Noisy `column` of `s`
        """

        self.__check_dataframe_initialized()
        # only for mypy: no effect on data/performance
        self.df = cast(pd.DataFrame, self.df)
        assert np.isscalar(s[column])

        # add noise
        noise: float = self.truncated_normal_noise(mean=mean, std=std,
            low=lb, upp=ub, size=1).item()  # must be ndim == 0
        s[column] = s[column] + noise
        return s


class AGE_CATEGORY(Enum):
    """Enumerator for categorizing based on age

    """
    KID = (1, 0, 12)
    TEEN = (2, 13, 20)
    YOUNG = (3, 20, 31)
    ADULT = (4, 31, 999)

    def __init__(self, id: int, start: float, end: float) -> None:
        """Each member of this class is a tuple of (`id`, lower bound, upper bound) of age range

        Args:
            id (int): ID for each Enum member
            start (float): Lower bound of age range of a category
            end (float): Upper bound of age range of a category
        """
        self.id = id
        self.start = start
        self.end = end

    @classmethod
    def categorize_age(self, age: float) -> AGE_CATEGORY:
        """takes an age and categorize it

        Args:
            age (float): input age to be categorized

        Raises:
            ValueError: If input is not in any of the categories.

        Returns:
            str: One of the following categories:

                * kid:    0 <= dob <  12
                * teen:  13 <= dob <  20
                * young: 20 <= dob <= 30
                * adult: 30 <  dob <  inf

        """
        if AGE_CATEGORY.KID.start <= age <  AGE_CATEGORY.KID.end:
            return AGE_CATEGORY.KID
        elif AGE_CATEGORY.TEEN.start <= age < AGE_CATEGORY.TEEN.end:
            return AGE_CATEGORY.TEEN
        elif AGE_CATEGORY.YOUNG.start <= age < AGE_CATEGORY.YOUNG.end:
            return AGE_CATEGORY.YOUNG
        elif AGE_CATEGORY.ADULT.start <= age:
            return AGE_CATEGORY.ADULT
        else:
            raise ValueError(f'"{age}" is not valid!')
