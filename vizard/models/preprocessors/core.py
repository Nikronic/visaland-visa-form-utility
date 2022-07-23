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
import pathlib
import json


# path to all config/db files
parent_dir = pathlib.Path(__file__).parent
DATA_DIR = parent_dir / 'data'

# configure logging
logger = logging.getLogger(__name__)


class TrainTestEvalSplit:
    """Convert a pandas dataframe to a numpy array for with train, test, and eval splits

    For conversion from :class:`pandas.DataFrame` to :class:`numpy.ndarray`, we use the same
    functionality as :class:`pandas.DataFrame.to_numpy`, but it separates dependent and
    independent variables given the target column ``target_column``.

    Note:
        To obtain the eval set, we use the train set as the original data to be splitted
        I.e. the eval set is a subset of train set.
        This is of course to make sure model by no means sees the test set.

    Note:
        ``Args`` cannot be set directly and need to be provided using a json file.
        See :meth:`set_configs` for more information.

    Note:
        You can explicitly override following attributes by passing it as an argument
        to :meth:`__init__`:
            - :attr:`random_state`
            - :attr:`stratify`

    Returns:
        Tuple[:class:`numpy.ndarray`, ...]: Order is
            ``(x_train, x_test, x_eval, y_train, y_test, y_eval)``
    """

    def __init__(self, stratify: Any = None,
                 random_state: Union[np.random.Generator, int] = None) -> None:
        self.logger = logging.getLogger(logger.name + self.__class__.__name__)
        self.CONF = self.set_configs()

        # override configs if explicitly set
        self.random_state = random_state
        self.stratify = stratify

    def set_configs(self, path: Union[str, pathlib.Path] = None) -> dict:
        """Defines and sets the config to be parsed

        The keys of the configs are the attributes of this class which are:
            test_ratio (float): Ratio of test data
            eval_ratio (float): Ratio of eval data
            shuffle (bool): Whether to shuffle the data
            stratify (Optional[np.ndarray]): If not None, this is used to stratify the data
            random_state (Optional[int]): Random state to use for shuffling

        Note:
            You can explicitly override following attributes by passing it as an argument
            to :meth:`__init__`:
                - :attr:`random_state`
                - :attr:`stratify`

        The values of the configs are parameters and can be set manually
        or extracted from JSON config files by providing the path to the JSON file.

        Args:
            path: path to the JSON file containing the configs

        Returns:
            dict: A dictionary of `str`: `Any` pairs of configs as class attributes
        """

        # convert str path to Path
        if isinstance(path, str):
            path = pathlib.Path(path)

        # if no json is provided, use the default configs
        if path is None:
            path = DATA_DIR / 'canada_train_test_eval_split.json'
        self.conf_path = path

        # log the path used
        self.logger.info(f'Config file "{self.conf_path}" is being used')

        # read the json file
        with open(path, 'r') as f:
            configs = json.load(f)

        # set the configs if explicit calls made to this method
        self.CONF = configs
        # return the parsed configs
        return configs

    def as_mlflow_artifact(self, target_path: Union[str, pathlib.Path]) -> None:
        """Saves the configs to the MLFlow artifact directory

        Args:
            target_path: Path to the MLFlow artifact directory. The name of the file
                will be same as original config file, hence, only provide path to dir.
        """

        # convert str path to Path
        if isinstance(target_path, str):
            target_path = pathlib.Path(target_path)

        if self.conf_path is None:
            raise ValueError(
                'Configs have not been set yet. Use set_configs to set them.')

        # read the json file
        with open(self.conf_path, 'r') as f:
            configs = json.load(f)

        # save the configs to the artifact directory
        target_path = target_path / self.conf_path.name
        with open(target_path, 'w') as f:
            json.dump(configs, f)

    def __call__(self, df: pd.DataFrame, target_column: str,
                 *args: Any, **kwds: Any) -> Tuple[np.ndarray, ...]:
        """Convert a pandas dataframe to a numpy array for with train, test, and eval splits
        
        Args:
            df (:class:`pandas.DataFrame`): Dataframe to convert
            target_column (str): Name of the target column
        
        Returns:
            Tuple[:class:`numpy.ndarray`, ...]: Order is
            ``(x_train, x_test, x_eval, y_train, y_test, y_eval)``
        """
        # get values from config
        test_ratio = self.CONF['test_ratio']
        eval_ratio = self.CONF['eval_ratio']
        shuffle = self.CONF['shuffle']
        stratify = self.CONF['stratify']
        random_state = self.CONF['random_state'] if self.random_state is None else self.random_state

        # separate dependent and independent variables
        y = df[target_column].to_numpy()
        x = df.drop(columns=[target_column], inplace=False).to_numpy()

        # create train and test data
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=None,
                                                                            test_size=test_ratio,
                                                                            shuffle=shuffle,
                                                                            stratify=stratify,
                                                                            random_state=random_state)
        if eval_ratio == 0.:
            return (x_train, x_test, y_train, y_test)

        # create eval data from train data
        x_train, x_eval, y_train, y_eval = model_selection.train_test_split(x_train, y_train,
                                                                            train_size=None,
                                                                            test_size=eval_ratio,
                                                                            shuffle=shuffle,
                                                                            stratify=stratify,
                                                                            random_state=random_state)
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


class ColumnSelector:
    """Selects columns based on regex pattern and dtype

    User can specify the dtype of columns to select, and the dtype of columns to ignore.
    Also, user can specify the regex pattern for including and excluding columns, separately.

    This is particularly useful when combined with :class:`sklearn.compose.ColumnTransformer`
    to apply different sort of ``transformers`` to different subsets of columns. E.g::

        # select columns that contain 'Country' in their name and are of type `np.float32`
        columns = preprocessors.ColumnSelector(columns_type='numeric',
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
        If the data that is passed to the :class:`ColumnSelector` is a :class:`pandas.DataFrame`,
        then you can ignore calling the instance of this class and directly use it in the
        pipeline. E.g::

            # select columns that contain 'Country' in their name and are of type `np.float32`
            columns = preprocessors.ColumnSelector(columns_type='numeric',
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
        :class:`sklearn.compose.make_column_selector` as ``ColumnSelector`` follows the
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

                1. ``'string'``: returns the name of the columns. Useful for 
                    :class:`pandas.DataFrame`
                2. ``'numeric'``: returns the index of the columns. Useful for
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


class ColumnTransformerConfig:
    """A helper class that parses configs for using the :class:`sklearn.compose.ColumnTransformer`

    The purpose of this class is to create the list of ``transformers`` to be used
    by the :class:`sklearn.compose.ColumnTransformer`. Hence, one needs to define the configs
    by using the :meth:`set_configs` method. Then use the :meth:`generate_pipeline` method
    to create the list of transformers.

    This class at the end, will return a list of tuples, where each tuple is a in
    the form of ``(name, transformer, columns)``.
    """

    def __init__(self) -> None:

        self.logger = logging.getLogger(logger.name+self.__class__.__name__)
        self.CONF = self.set_configs()

    def set_configs(self, path: Union[str, pathlib.Path] = None) -> dict:
        """Defines and sets the config to be parsed

        The keys of the configs are the names of the transformers. They must include
        one of the following at the end:

            - ``'categorical'``: to be used with :class:`sklearn.preprocessing.OneHotEncoder`
            or any other function that processes *categorical* data
            - ``'continuous'``: to be used with :class:`sklearn.preprocessing.StandardScaler`
            or any other function that processes *continuous* data
            - ``'binary'``: to be used with :class:`sklearn.preprocessing.LabelEncoder`
            or any other function that processes *binary* data

        This naming convention is used to create proper transformers for each type of data.


        The values of the configs are the columns to be transformed. The columns can be
        obtained by using :class:`vizard.models.preprocessors.core.ColumnSelector`
        which requires user to pass certain parameters. This parameters can be set manually
        or extracted from JSON config files by providing the path to the JSON file.

        Args:
            path: path to the JSON file containing the configs

        Returns:
            dict: A dictionary of `str`: :class:`vizard.models.preprocessors.core.ColumnSelector`
            which will be passed to :meth:`generate_pipeline`
        """

        # convert str path to Path
        if isinstance(path, str):
            path = pathlib.Path(path)

        # if no json is provided, use the default configs
        if path is None:
            path = DATA_DIR / 'canada_column_transformer_config.json'
        self.conf_path = path

        # log the path used
        self.logger.info(f'Config file "{self.conf_path}" is being used')

        # read the json file
        with open(path, 'r') as f:
            configs = json.load(f)

        # parse the configs (json files)
        parsed_configs = {}
        for key, value in configs.items():
            parsed_values = {k: eval(v) for k, v in value.items()}
            parsed_configs[key] = ColumnSelector(**parsed_values)

        # set the configs when explicit call to this method is made
        self.CONF = parsed_configs

        # return the parsed configs
        return parsed_configs

    def as_mlflow_artifact(self, target_path: Union[str, pathlib.Path]) -> None:
        """Saves the configs to the MLFlow artifact directory

        Args:
            target_path: Path to the MLFlow artifact directory. The name of the file
                will be same as original config file, hence, only provide path to dir.
        """

        # convert str path to Path
        if isinstance(target_path, str):
            target_path = pathlib.Path(target_path)

        if self.conf_path is None:
            raise ValueError(
                'Configs have not been set yet. Use set_configs to set them.')

        # read the json file
        with open(self.conf_path, 'r') as f:
            configs = json.load(f)

        # save the configs to the artifact directory
        target_path = target_path / self.conf_path.name
        with open(target_path, 'w') as f:
            json.dump(configs, f)

    @staticmethod
    def extract_selected_columns(selector: ColumnSelector,
                                 df: pd.DataFrame) -> Union[List[str], List[int]]:
        """Extracts the columns from the dataframe based on the selector

        Note:
            This method is simply a wrapper around :class:`vizard.models.preprocessors.core.ColumnSelector`
            that makes the call given a dataframe. I.e.::

                # assuming same configs
                selector = preprocessors.ColumnSelector(...)
                A = ColumnTransformerConfig.extract_selected_columns(selector=selector, df=df)
                B = selector(df)
                A == B  # True

            Also, this is a static method.

        Args:
            selector (:class:`vizard.models.preprocessors.core.ColumnSelector`): Initialized
                selector object
            df (pd.DataFrame): Dataframe to extract columns from

        Returns:
            Union[List[str], List[int]]: List of columns to be transformed
        """
        return selector(df=df)

    def generate_pipeline(self, df: pd.DataFrame) -> list:
        """Generates the list of transformers to be used by the :class:`sklearn.compose.ColumnTransformer`

        Note:
            For more info about how the transformers are created, see methods
            :meth:`set_configs` and :meth:`extract_selected_columns`.

        Args:
            df (pd.DataFrame): Dataframe to extract columns from

        Raises:
            ValueError: If the naming convention used for the keys in the
                configs (see :meth:`set_configs`) is not followed.

        Returns:
            list: A list of tuples, where each tuple is a in the form of
            ``(name, transformer, columns)`` where ``name`` is the name of the
            transformer, ``transformer`` is the transformer object and ``columns``
            is the list of columns (``List[str]``) to be transformed.
        """

        # just place holders for what we want
        name: str = ''              # name of the transformer
        transformer: object = None  # transformer object
        columns: List = []          # columns to transform

        # list of (name, transformer, columns) tuples to return
        transformers: List[Tuple] = []

        # iterate through the configs to build transformer instances appropriately
        for key, value in self.CONF.items():
            # if categorical, use OneHotEncoder
            if 'categorical' in key:
                name = key
                transformer = OneHotEncoder()
                columns = self.extract_selected_columns(selector=value, df=df)
            # if continuous, use StandardScaler
            elif 'continuous' in key:
                name = key
                transformer = StandardScaler()
                columns = self.extract_selected_columns(selector=value, df=df)
            # if binary, use LabelEncoder
            elif 'binary' in key:
                name = key
                transformer = LabelEncoder()
                columns = self.extract_selected_columns(selector=value, df=df)
            # if other, raise exception
            else:
                raise ValueError(
                    f'Unknown dtype type for "key:value" config: {key}:{value}')

            # add to the list of transformers
            transformers.append((name, transformer, columns))

        return transformers
