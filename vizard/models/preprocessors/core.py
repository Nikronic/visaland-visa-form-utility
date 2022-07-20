"""Contains core functionalities that is shared by all preprocessors.

"""

# core
from sklearn import model_selection
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import numpy as np
# helpers
from typing import Tuple, Optional, Any, List, Union
import logging


# configure logging
logger = logging.getLogger(__name__)


def train_test_eval_split(df: pd.DataFrame, target_column: str,
                          test_ratio: float = 0.0,
                          eval_ratio: float = 0.0,
                          **kwargs) -> Tuple[np.ndarray, ...]:
    """Convert a pandas dataframe to a numpy array for with train, test, and eval splits

    For conversion from :class:`pandas.DataFrame` to :class:`numpy.ndarray`, we use the same
    functionality as :class:`pandas.DataFrame.to_numpy`, but it separates dependent and
    independent variables given the target column ``target_column``.

    Note:
        To obtain the eval set, we use the train set as the original data to be splitted
        I.e. the eval set is a subset of train set.
        This is of course to make sure model by no means sees the test set.

    Args:
        df (:class:`pandas.DataFrame`): Dataframe to convert
        target_column (str): Name of the target column
        test_ratio (float): Ratio of test data
        eval_ratio (float): Ratio of eval data
        shuffle (bool): Whether to shuffle the data
        stratify (Optional[np.ndarray]): If not None, this is used to stratify the data
        random_state (Optional[int]): Random state to use for shuffling

    Returns:
        Tuple[:class:`numpy.ndarray`, ...]: Order is
            ``(x_train, x_test, x_eval, y_train, y_test, y_eval)``

    """

    # separate dependent and independent variables
    y = df[target_column].to_numpy()
    x = df.drop(columns=[target_column], inplace=False).to_numpy()

    # create train and test data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=None,
                                                                        test_size=test_ratio,
                                                                        **kwargs)
    if eval_ratio == 0.:
        return (x_train, x_test, y_train, y_test)

    # create eval data from train data
    x_train, x_eval, y_train, y_eval = model_selection.train_test_split(x_train, y_train,
                                                                        train_size=None,
                                                                        test_size=eval_ratio,
                                                                        **kwargs)
    return (x_train, x_test, x_eval, y_train, y_test, y_eval)


def move_dependent_variable_to_end(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Move the dependent variable to the end of the dataframe

    This is useful for some frameworks that require the dependent variable to be the last
    or in general form, it is way easier to play with :class:`numpy.ndarray`s when the
    dependent variable is the last one.

    Note:
        This is particularly is useful for us since we have multiple columns of the same
        type in our dataframe, and when we want to apply same preprocessing to a all members
        of a group of features, we can directly use index of those features from our pandas
        dataframe in converted numpy array. E.g::

            df = pd.DataFrame(...)
            x = df.to_numpy()
            index = df.columns.get_loc(a_group_of_columns_with_the_same_logic)
            x[:, index] = transform(x[:, index])

    Args:
        df (:class:`pandas.DataFrame`): Dataframe to convert
        target_column (str): Name of the target column

    Returns:
        :class:`pandas.DataFrame`: Dataframe with the dependent variable at the end

    """

    columns = df.columns.tolist()
    columns.pop(columns.index(target_column))
    df = df[columns + [target_column]]
    return df


class column_selector:
    """Selects columns based on regex pattern and dtype

    User can specify the dtype of columns to select, and the dtype of columns to ignore.
    Also, user can specify the regex pattern for including and excluding columns, separately.

    This is particularly useful when combined with :class:`sklearn.compose.ColumnTransformer`
    to apply different sort of ``transformers`` to different subsets of columns. E.g::

        # select columns that contain 'Country' in their name and are of type `np.float32`
        columns = preprocessors.column_selector(columns_type='numeric',
                                                dtype_include=np.float32,
                                                pattern_include='.*Country.*',
                                                pattern_exclude=None,
                                                dtype_exclude=None)(df=data)
        # use a transformer for selected columns
        ct = preprocessors.ColumnTransformer(
            [('some_name',                   # just a name
            preprocessors.StandardScaler(),  # the transformer
            columns),                        # the columns to apply the transformer to
            ],
        )

        ct.fit_transform(...)

    Note:
        If the data that is passed to the :class:`column_selector` is a :class:`pandas.DataFrame`,
        then you can ignore calling the instance of this class and directly use it in the
        pipeline. E.g::

            # select columns that contain 'Country' in their name and are of type `np.float32`
            columns = preprocessors.column_selector(columns_type='numeric',
                                                    dtype_include=np.float32,
                                                    pattern_include='.*Country.*',
                                                    pattern_exclude=None,
                                                    dtype_exclude=None)  # THIS LINE
            # use a transformer for selected columns
            ct = preprocessors.ColumnTransformer(
                [('some_name',                   # just a name
                preprocessors.StandardScaler(),  # the transformer
                columns),                        # the columns to apply the transformer to
                ],
            )

            ct.fit_transform(...)


    See Also:
        :class:`sklearn.compose.make_column_selector` as ``column_selector`` follows the
        same semantics.

    """

    def __init__(self,
                 columns_type: str,
                 dtype_include: Any,
                 pattern_include: Optional[str] = None,
                 dtype_exclude: Any = None,
                 pattern_exclude: Optional[str] = None) -> None:
        """Selects columns based on regex pattern and dtype

        Args:
            columns_type (str): Type of columns:

                1. 'string': returns the name of the columns. Useful for 
                    :class:`pandas.DataFrame`
                2. 'numeric': returns the index of the columns. Useful for
                    :class:`numpy.ndarray`

            dtype_include (type): Type of the columns to select. For more info
                see :func:`pandas.DataFrame.select_dtypes`.
            pattern_include (str): Regex pattern to match columns to **include**
            dtype_exclude (type): Type of the columns to ignore. For more info
                see :func:`pandas.DataFrame.select_dtypes`. Defaults to None.
            pattern_exclude (str): Regex pattern to match columns to **exclude**
        """
        self.columns_type = columns_type
        self.pattern_include = pattern_include
        self.pattern_exclude = pattern_exclude
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, df: pd.DataFrame,
                 *args: Any, **kwds: Any) -> Union[List[str], List[int]]:
        """

        Args:
            df (:class:`pandas.DataFrame`): Dataframe to extract columns from

        Returns:
            Union[List[str], List[int]]: List of names or indices of
                filtered columns

        Raises:
            ValueError: If the ``df`` is not instance of :class:`pandas.DataFrame`

        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'`df` must be a `DataFrame` not {type(df)}')

        # since `make_column_selector` will ignore pattern if None provide,
        # we need to set pattern (pattern_exclude) to sth pepega to ignore all columns
        pattern_exclude = '\~' if self.pattern_exclude is None else self.pattern_exclude

        # first select desired then select undesired
        columns_to_include = make_column_selector(dtype_include=self.dtype_include,
                                                  dtype_exclude=self.dtype_exclude,
                                                  pattern=self.pattern_include)(df)
        columns_to_exclude = make_column_selector(dtype_include=self.dtype_include,
                                                  dtype_exclude=self.dtype_exclude,
                                                  pattern=pattern_exclude)(df)

        # remove columns_to_exclude from columns_to_include
        columns = [
            column for column in columns_to_include if column not in columns_to_exclude]

        # return columns based on columns_type (`columns` is already `string`)
        if self.columns_type == 'numeric':
            return [df.columns.get_loc(column) for column in columns]
        return columns
