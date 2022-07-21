# include all libs that are used in DATA/*.json files
import numpy as np
# core
import pandas as pd
import json
# ours
from vizard.models import preprocessors
# helpers
import pathlib
import logging
from typing import List, Tuple, Union


# path to all config/db files
parent_dir = pathlib.Path(__file__).parent
DATA_DIR = parent_dir / 'data'

# configure logging
logger = logging.getLogger(__name__)


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

            - 'categorical': to be used with :class:`sklearn.preprocessing.OneHotEncoder`
            or any other function that processes categorical data
            - 'continuous': to be used with :class:`sklearn.preprocessing.StandardScaler`
            or any other function that processes continuous data
            - 'binary': to be used with :class:`sklearn.preprocessing.LabelEncoder`
            or any other function that processes binary data
        
        This naming convention is used to create proper transformers for each type of data.


        The values of the configs are the columns to be transformed. The columns can be
        obtained by using :class:`vizard.models.preprocessors.core.column_selector`
        which requires user to pass certain parameters. This parameters can be set manually
        or extracted from JSON config files by providing the path to the JSON file.

        Args:
            path: path to the JSON file containing the configs

        Returns:
            dict: A dictionary of `str`: :class:`vizard.models.preprocessors.core.column_selector`
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
            parsed_configs[key] = preprocessors.column_selector(**parsed_values)

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
            raise ValueError('Configs have not been set yet. Use set_configs to set them.')

        # read the json file
        with open(self.conf_path, 'r') as f:
            configs = json.load(f)

        # save the configs to the artifact directory
        target_path = target_path / self.conf_path.name
        with open(target_path, 'w') as f:
            json.dump(configs, f)
    
    @staticmethod
    def extract_selected_columns(selector: preprocessors.column_selector,
                                 df: pd.DataFrame) -> Union[List[str], List[int]]:
        """Extracts the columns from the dataframe based on the selector

        Note:
            This method is simply a wrapper around :class:`vizard.models.preprocessors.core.column_selector`
            that makes the call given a dataframe. I.e.::

                # assuming same configs
                selector = preprocessors.column_selector(...)
                A = ColumnTransformerConfig.extract_selected_columns(selector=selector, df=df)
                B = selector(df)
                A == B  # True
            
            Also, this is a static method.

        Args:
            selector (:class:`vizard.models.preprocessors.core.column_selector`): Initialized
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
                transformer = preprocessors.OneHotEncoder()
                columns = self.extract_selected_columns(selector=value, df=df)
            # if continuous, use StandardScaler
            elif 'continuous' in key:
                name = key
                transformer = preprocessors.StandardScaler()
                columns = self.extract_selected_columns(selector=value, df=df)
            # if binary, use LabelEncoder
            elif 'binary' in key:
                name = key
                transformer = preprocessors.LabelEncoder()
                columns = self.extract_selected_columns(selector=value, df=df)
            # if other, raise exception
            else:
                raise ValueError(f'Unknown dtype type for "key:value" config: {key}:{value}')
            
            # add to the list of transformers
            transformers.append((name, transformer, columns))
        
        return transformers
                