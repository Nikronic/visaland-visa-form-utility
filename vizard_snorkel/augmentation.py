from typing import Any, Optional, Tuple, Union, cast
from snorkel.augmentation.tf import transformation_function
from snorkel.augmentation.tf import TransformationFunction
import numpy as np
import pandas as pd


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
        """
        A wrapper around `Numpy.Generator.normal`. 

        Dev may extend this method by adding features to it; e.g. adding it to 
            a pandas Series (see `self.add_normal_noise`)

        """
        return self.rng.normal(loc=mean, scale=std, size=size)

    def series_add_normal_noise(self, s: pd.Series, column: str) -> pd.Series:
        """
        Takes a pandas Series and corresponding column and adds *normal* noise
            to it with mean and std obtained from original dataframe[column]
            that `s` extracted from 

        args:
            s: Pandas Series to be manipulated (from `self.df`)
            column: corresponding column in `self.df` and `s` 
        """
        self.__check_dataframe_initialized()
        self.df = cast(pd.DataFrame, self.df)  # only for mypy: no effect on data/performance
        assert np.isscalar(s[column])

        mean, std = self.df[column].mean(), self.df[column].std()
        noise: float = self.normal_noise(
            mean=mean, std=std, size=1).item()  # must be ndim == 0
        s[column] = s[column] + noise
        return s

# init an instance of `SeriesNoise` to use its functions as `TransformationFunction`
series_noise_utils = SeriesNoise(dataframe=None)

def make_add_normal_noise_tf(column: str) -> TransformationFunction:
    """
    A helper wrapper around `TransformationFunction`, here specifically for
        `SeriesNoise.series_add_normal_noise` functions that will be only called
        over continuous columns of the series `s` which can be manipulated using
        given function `f` (here `=series_add_normal_noise`)
    
    """
    return TransformationFunction(
        name=f'add_normal_noise_{column}',
        f=series_noise_utils.series_add_normal_noise,
        resources=dict(column=column),
    )


Funds_lf = make_add_normal_noise_tf('P3.DOV.PrpsRow1.Funds.Funds')
DOBYear = make_add_normal_noise_tf('P1.PD.DOBYear.Period')
