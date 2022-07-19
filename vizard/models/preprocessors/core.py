"""Contains core functionalities that is shared by all preprocessors.

"""

# core
from sklearn import model_selection
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
