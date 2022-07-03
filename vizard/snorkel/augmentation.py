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


class SeriesNoise:
    """
    Adds different type noise (multiplicative or additive) to a `Pandas.Series`

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
