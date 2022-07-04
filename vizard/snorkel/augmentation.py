# for forward reference https://stackoverflow.com/a/50768146/18971263
from __future__ import annotations


__all__ = [
    'SeriesNoise', 'TFAugmentation', 'ComposeTFAugmentation', 
    'AddNormalNoiseDOBYear', 'AGE_CATEGORY', 'AddNormalNoiseFunds'
]

# core
from scipy.stats import truncnorm
import pandas as pd
import numpy as np
# snorkel
from snorkel.augmentation import transformation_function
from snorkel.augmentation import TransformationFunction
# helpers
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast
from enum import Enum


class SeriesNoise:
    """
    Adds different type noise (impulse, multiplicative, or additive) to a `Pandas.Series`

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

    def categorical_switch_noise(self, s: pd.Series, column: str,
                                 categories: dict, **kwargs) -> pd.Series:
        """Takes a pandas Series and corresponding column and switches it with uniform distribution

        Args:
            s (pd.Series): Pandas Series to be manipulated (from `self.df`)
            column (str): corresponding column in `self.df` and `s` 
            categories (dict): dictionary of categories to switch to
            kwargs (dict): keyword arguments for [numpy.random.Generator.choice]_

        Returns:
            pd.Series: Categorically shuffled `column` of `s`
        
        .. [#] https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html

        """
        self.__check_dataframe_initialized()
        self.df = cast(pd.DataFrame, self.df)
        if s[column] not in categories.keys():
            raise ValueError(f'{s[column]} not in {categories}')
        # TODO: do a proper checking here, if bad input, raise valueerror exception
        

        if s[column] in categories[s[column]]:
            raise ValueError((f'key "{s[column]}" cannot also exist in',
                              f'its values "{categories[s[column]]}"'))
        # switch
        s[column] = self.rng.choice([categories[s[column]]], **kwargs)
        return s


class TFAugmentation:
    """Adds augmentation capabilities unique to a country for `snorkel.TransformationFunction`

    Notes:
        User must create new class that subclasses this and write domain/dataset
            specific methods for his/her case. For instance, if you need to add
            augmentation to a class with "age" value, then extend this class,
            add new method e.g. ``[add_noise_]age``.
        
        At the moment, the goal is to use this base to include all core augmentation
            methods that could be used anywhere but subclassing them,
            such as continuous noises, shuffling, etc.
            And make sure that any class that subclass this, should integrate 
            `snorkel.TransformationFunction` to be usable in `snorkel` pipeline.
            See `CanadaTFAugmentation` for instance of implementation and usage.

    """

    def __init__(self) -> None:
        self.COLUMN = ''

    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augments a Pandas Series by modifying a single column

        Args:
            s (pd.Series): Pandas Series to be processed

        Returns:
            pd.Series: Augmented `s` on column `COLUMN`
        """
        raise NotImplementedError

    def make_tf(self, func: Callable,
                class_name: Optional[str] = None,
                **kwargs) -> TransformationFunction:
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
            class_name (str, optional): The name of the class if `func` is a method of it.
                It is used for better naming given class name alongside `func` name.
                Defaults to None.
            kwargs: keyword arguments to `func`

        Returns:
            TransformationFunction: A callable object that is now compatible with
                `snorkel` transformation pipeline, e.g. ``Policy`` and ``TFApplier``
        """

        column = kwargs.pop('column')

        if class_name is None:
            return TransformationFunction(
                name=f'{func.__name__}_{column}',
                f=func, resources=dict(column=column, **kwargs),
            )
        else:
            return TransformationFunction(
                name=f'{class_name}_{column}',
                f=func, resources=dict(column=column, **kwargs),
            )    


class ComposeTFAugmentation(TFAugmentation):
    """Composes a list of `TFAugmentation` instances 

    Examples:
        >>> tf_compose = [
        >>>     augmentation.AddNormalNoiseDOBYear(dataframe=data)
        >>> ]
        >>> tfs = augmentation.ComposeTFAugmentation(augments=tf_compose)()
        >>> tf_applier = PandasTFApplier(tfs, random_policy)

    """
    def __init__(self, augments: Sequence[TFAugmentation]) -> None:
        super().__init__()

        for aug in augments:
            if not issubclass(aug.__class__, TFAugmentation):
                raise TypeError(
                    'Keys must be {} instance.'.format(TFAugmentation))

        self.augments = augments
    
    def __call__(self, *args: Any, **kwds: Any) -> list[TransformationFunction]:
        """Takes a list of `TFAugmentation` and converts to `snorkel.TransformationFunction`

        Returns:
            list[TransformationFunction]: A list of objects that instantiate
                `snorkel.TransformationFunction` 
        """
        augments_tf: list[TransformationFunction] = []
        for aug in self.augments:
            aug = self.make_tf(func=aug.augment, column=aug.COLUMN,
                               class_name=aug.__class__.__name__)
            augments_tf.append(aug)
        return augments_tf


class AddNormalNoiseDOBYear(SeriesNoise, TFAugmentation):
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__decay = 0.9
        self.COLUMN = 'P1.PD.DOBYear.Period'
        self.__std = 2. * self.__decay
        # neighborhood for truncated normal filled with shortest period
        __max_bound = AGE_CATEGORY.TEEN.end - AGE_CATEGORY.TEEN.start
        self.__max_bound = __max_bound * self.__decay

    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Add normal noise to ``'P1.PD.DOBYear.Period'``

        This methods makes sure that by adding noise, the age does not
            fall into a new category. See `categorize_age` for more info.
        In other words, we make sure a normal noise is defined within range of
            each category, hence always noisy data will stay in same category.

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy ``'P1.PD.DOBYear.Period'`` of `s`
        """

        COLUMN = self.COLUMN
        # construct normal distribution over neighborhood around input
        lower_bound = -(s[COLUMN] - self.__max_bound)  # can only be <= 0
        upper_bound = (s[COLUMN] - self.__max_bound)    # can only be >= 0
        s = self.series_add_truncated_normal_noise(s=s, column=COLUMN, 
                                                   mean=0., std=self.__std,
                                                   lb=lower_bound, ub=upper_bound)
        return s


class AddNormalNoiseFunds(SeriesNoise, TFAugmentation):
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__decay = [0.2, 0.1]
        self.COLUMN = 'P3.DOV.PrpsRow1.Funds.Funds'

    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Add normal noise to ``'P3.DOV.PrpsRow1.Funds.Funds'``

        Following conditions have been used:
            1. ``'hasInvLttr'``: we can choose larger neighborhood if this is True. 
                The decay percentage can be found in `__decay`.

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy ``'P3.DOV.PrpsRow1.Funds.Funds'`` of `s`
        """

        # TODO: add fin.bb to make sure by adding more fund, it does not any issue with
        #   fin.bb value, i.e. cap the positive addition value if it was too much
        # fixed column corresponding to this function
        COLUMN = self.COLUMN
        COND_COLUMN = ['hasInvLttr']
        # values to add noise based on a categorization
        decay = {
            True: self.__decay[0],
            False: self.__decay[1],
        }
        has_inv_letter = s[COND_COLUMN[0]]  # condition
        s = self.series_add_normal_noise(s=s, column=COLUMN, mean=0., 
                                         std=s[COLUMN] * decay[has_inv_letter])
        return s


class AddNormalNoiseDateOfMarr(SeriesNoise, TFAugmentation):
    """Add normal noise to ``'P1.MS.SecA.DateOfMarr.Period'``

    Entries where value of `column` in `s` is zero will be ignored. I.e.
        those who are "single" would stay single where "single" means
        "non-married" and "previously married"

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__decay = 0.2
        self.COLUMN = 'P1.MS.SecA.DateOfMarr.Period'
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """
        COLUMN = self.COLUMN
        if s[COLUMN] != 0.:
            s = self.series_add_normal_noise(s=s, column=COLUMN, mean=0., 
                                             std=s[COLUMN] * self.__decay)
        return s


class AddNormalNoiseOccRow1Period(SeriesNoise, TFAugmentation):
    """Add normal noise to ``'P3.Occ.OccRow1.Period'``

    Entries where value of `column` in `s` is zero will be ignored. I.e.
        those who are "uneducated" would stay educated.

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__decay = 0.2
        self.COLUMN = 'P3.Occ.OccRow1.Period'
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN
        std = s[COLUMN] * self.__decay
        ub = std
        lb = -ub

        if s[COLUMN] != 0.:
            s = self.series_add_truncated_normal_noise(s=s, column=COLUMN,
                                                       mean=0., std=std,
                                                       lb=lb, ub=ub)
        return s


class AddNormalNoiseOccRow2Period(SeriesNoise, TFAugmentation):
    """Add normal noise to ``'P3.Occ.OccRow2.Period'``

    Entries where value of `column` in `s` is zero will be ignored. I.e.
        those who are "uneducated" would stay educated.

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__decay = 0.2
        self.COLUMN = 'P3.Occ.OccRow2.Period'
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN
        std = s[COLUMN] * self.__decay
        ub = std
        lb = -ub

        if s[COLUMN] != 0.:
            s = self.series_add_truncated_normal_noise(s=s, column=COLUMN,
                                                       mean=0., std=std,
                                                       lb=lb, ub=ub)
        return s


class AddNormalNoiseOccRow3Period(SeriesNoise, TFAugmentation):
    """Add normal noise to ``'P3.Occ.OccRow3.Period'``

    Entries where value of `column` in `s` is zero will be ignored. I.e.
        those who are "uneducated" would stay educated.

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__decay = 0.2
        self.COLUMN = 'P3.Occ.OccRow3.Period'
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN
        std = s[COLUMN] * self.__decay
        ub = std
        lb = -ub

        if s[COLUMN] != 0.:
            s = self.series_add_truncated_normal_noise(s=s, column=COLUMN,
                                                       mean=0., std=std,
                                                       lb=lb, ub=ub)
        return s


class AddNormalNoiseHLS(SeriesNoise, TFAugmentation):
    """Add normal noise to ``'P3.DOV.PrpsRow1.HLS.Period'``

    Entries where value of `column` in `s` is below 14 (2 weeks)
        only will receive truncated noise with positive value. Also,
        no value after getting noisy could be under 14. In simple terms, 
        all values have to be above 14.
    I.e. (conditions):

        * if below 14 (or smaller) -> just add + noise
        * if close 14 -> make sure dont go below 14
        * if above 21 -> free to do anything

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        # values to add noise based on a categorization
        self.__std = 7  # we can do +- 7 days 
        self.COLUMN = 'P3.DOV.PrpsRow1.HLS.Period'
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN

        if (s[COLUMN] > 14.) and (s[COLUMN] < 21.):  # ~75% of entries
            # add [-[0, 7], 7] days
            s = self.series_add_truncated_normal_noise(s=s, column=COLUMN, 
                                                   mean=0., std=self.__std,
                                                   lb=-(21. - s[COLUMN]), ub=7.)
        elif s[COLUMN] >= 21.:
            # add [-7, 7] days
            s = self.series_add_truncated_normal_noise(s=s, column=COLUMN, 
                                                   mean=0., std=self.__std,
                                                   lb=-7., ub=7.)
        elif s[COLUMN] <= 14.:
            # add [0, 7] days
            s = self.series_add_truncated_normal_noise(s=s, column=COLUMN, 
                                                   mean=0., std=self.__std,
                                                   lb=0., ub=7.)
        
        # must remove float part (must be in 'days')
        s[COLUMN] = np.int32(s[COLUMN])
        return s


class AddCategoricalNoiseChildRel0(SeriesNoise, TFAugmentation):
    """Add categorical noise to ``'p1.SecB.Chd.[0].ChdRel'``

    Entries where child exists (i.e. != 'other'), will be shuffled
        randomly based on Bernoulli trial. Note that it only changes
        the gender not relation level. Possible cases:

            * 'son' -> 'daughter'
            * 'step son' -> 'step daughter'
            * 'daughter' -> 'son'
            * 'step daughter' -> 'step son'

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        self.COLUMN = 'p1.SecB.Chd.[0].ChdRel'
        self.CATEGORIES = {
            'son': 'daughter',
            'step son': 'step daughter',
            'daughter': 'son',
            'step daughter': 'step son',
            'other': 'other'
        }
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN

        if s[COLUMN] != 'other':  # if child exists
            s = self.categorical_switch_noise(s=s, column=COLUMN,
                                              categories=self.CATEGORIES)
        return s


class AddCategoricalNoiseChildRel1(SeriesNoise, TFAugmentation):
    """Add categorical noise to ``'p1.SecB.Chd.[1].ChdRel'``

    Entries where child exists (i.e. != 'other'), will be shuffled
        randomly based on Bernoulli trial. Note that it only changes
        the gender not relation level. Possible cases:

            * 'son' -> 'daughter'
            * 'step son' -> 'step daughter'
            * 'daughter' -> 'son'
            * 'step daughter' -> 'step son'

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        self.COLUMN = 'p1.SecB.Chd.[1].ChdRel'
        self.CATEGORIES = {
            'son': 'daughter',
            'step son': 'step daughter',
            'daughter': 'son',
            'step daughter': 'step son',
            'other': 'other'
        }
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN

        if s[COLUMN] != 'other':  # if child exists
            s = self.categorical_switch_noise(s=s, column=COLUMN,
                                              categories=self.CATEGORIES)
        return s


class AddCategoricalNoiseChildRel2(SeriesNoise, TFAugmentation):
    """Add categorical noise to ``'p1.SecB.Chd.[2].ChdRel'``

    Entries where child exists (i.e. != 'other'), will be shuffled
        randomly based on Bernoulli trial. Note that it only changes
        the gender not relation level. Possible cases:

            * 'son' -> 'daughter'
            * 'step son' -> 'step daughter'
            * 'daughter' -> 'son'
            * 'step daughter' -> 'step son'

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        self.COLUMN = 'p1.SecB.Chd.[2].ChdRel'
        self.CATEGORIES = {
            'son': 'daughter',
            'step son': 'step daughter',
            'daughter': 'son',
            'step daughter': 'step son',
            'other': 'other'
        }
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN

        if s[COLUMN] != 'other':  # if child exists
            s = self.categorical_switch_noise(s=s, column=COLUMN,
                                              categories=self.CATEGORIES)
        return s

class AddCategoricalNoiseChildRel3(SeriesNoise, TFAugmentation):
    """Add categorical noise to ``'p1.SecB.Chd.[3].ChdRel'``

    Entries where child exists (i.e. != 'other'), will be shuffled
        randomly based on Bernoulli trial. Note that it only changes
        the gender not relation level. Possible cases:

            * 'son' -> 'daughter'
            * 'step son' -> 'step daughter'
            * 'daughter' -> 'son'
            * 'step daughter' -> 'step son'

    """
    def __init__(self, dataframe: Optional[pd.DataFrame]) -> None:
        super().__init__(dataframe)

        self.COLUMN = 'p1.SecB.Chd.[3].ChdRel'
        self.CATEGORIES = {
            'son': 'daughter',
            'step son': 'step daughter',
            'daughter': 'son',
            'step daughter': 'step son',
            'other': 'other'
        }
    
    def augment(self, s: pd.Series, column: str = None) -> pd.Series:
        """Augment the series for the predetermined column

        Args:
            s (pd.Series): A pandas series to get noisy on a fixed column

        Returns:
            pd.Series: Noisy `self.COLUMN` of `s`
        """

        COLUMN = self.COLUMN

        if s[COLUMN] != 'other':  # if child exists
            s = self.categorical_switch_noise(s=s, column=COLUMN,
                                              categories=self.CATEGORIES)
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
