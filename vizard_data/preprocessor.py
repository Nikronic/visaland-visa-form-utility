__all__ = ['DataframePreprocessor', 'CanadaDataframePreprocessor', 'UnitConverter',
           'FinancialUnitConverter', 'T0', 'FileTransformCompose', 'FileTransform', 'CopyFile',
           'MakeContentCopyProtectedMachineReadable']

import shutil
import pikepdf
import pandas as pd
import numpy as np
from dateutil import parser
from dateutil.relativedelta import *
from typing import Callable, List, Optional, Tuple, Union, Any
import logging

from vizard_data import functional
from vizard_data.constant import *
from vizard_data.PDFIO import CanadaXFA
from vizard_utils.helpers import loggingdecorator


# logging
logger = logging.getLogger(__name__)

T0 = '19000202T000000'  # a default meaningless time to fill the `None`s


class DataframePreprocessor:
    """
    A class that contains methods for dealing with dataframes regarding transformation of data
        such as filling missing values, dropping columns, or aggregating multiple
        columns into a single more meaningful one.
    This class needs to be extended for file specific preprocessing where tags are unique and need
        to be done entirely manually. In this case, `file_specific_preprocessor` needs to be implemented.
    """

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        self.dataframe = dataframe
        self.logger = logging.getLogger(logger.name+'.DataframePreprocessor')

    @loggingdecorator(logger.name+'.DataframePreprocessor.func', level=logging.DEBUG, output=False, input=True)
    def column_dropper(self, string: str, exclude: str = None, regex: bool = False,
                       inplace: bool = True) -> Union[None, pd.DataFrame]:
        """
        Takes a Pandas Dataframe and searches for columns *containing* `string` in them either 
            raw string or regex (in latter case, use `regex=True`) and after `exclude`ing a
            subset of them, drops the remaining *in-place*.

        args:
            dataframe: Pandas dataframe to be processed
            string: string to look for in `dataframe` columns
            exclude: string to exclude a subset of columns from being dropped 
            regex: compile `string` as regex
            inplace: whether or not use and inplace operation
        """

        return functional.column_dropper(dataframe=self.dataframe, string=string,
                                         exclude=exclude, regex=regex, inplace=inplace)

    @loggingdecorator(logger.name+'.DataframePreprocessor.func', level=logging.DEBUG, output=False)
    def fillna_datetime(self, col_base_name: str, type: DOC_TYPES, one_sided: Union[str, bool],
                        date: str = None, inplace: bool = False) -> Union[None, pd.DataFrame]:
        """
        In a Pandas Dataframe, takes two columns of dates in string form that has no value (None)
            and sets them to the same date which further ahead, in transformation operations
            such as `aggregate_datetime` function, it would be converted to period of zero.

        args:
            dataframe: Pandas Dataframe to be processed
            col_base_name: Base column name that accepts `From` and `To` for
                extracting dates of same category
            date: The desired date
            one_sided: Different ways of filling empty date columns:\n
                1. `'right'`: Uses the `current_date` as the final time
                2. `'left'`: Uses the `reference_date` as the starting time
            inplace: whether or not use and inplace operation
        """

        if date is None:
            date = T0

        return functional.fillna_datetime(dataframe=self.dataframe,
                                          col_base_name=col_base_name, one_sided=one_sided,
                                          date=date, inplace=inplace, type=type)

    @loggingdecorator(logger.name+'.DataframePreprocessor.func', level=logging.DEBUG, output=False)
    def aggregate_datetime(self, col_base_name: str, new_col_name: str,
                           type: DOC_TYPES, if_nan: Union[str, Callable, None] = None,
                           one_sided: str = None, reference_date: str = None,
                           current_date: str = None) -> pd.DataFrame:
        """
        In a Pandas Dataframe, takes two columns of dates in string form and calculates
            the period of these two dates and represent it in integer form. The two columns
            used will be dropped.

        E.g.:
        ```
            *.FromDate and *.ToDate --> *.Period | *.FromYear and *.ToYear --> *.Period in days
        ```
        args:
            dataframe: Pandas dataframe to be processed
            col_base_name: Base column name that accepts `From` and `To` for
                extracting dates of same category
            new_col_name: The column name that extends `col_base_name` and will be
                the final column containing the period.
            one_sided: Different ways of filling empty date columns:\n
                1. `'right'`: Uses the `current_date` as the final time
                2. `'left'`: Uses the `reference_date` as the starting time
            reference_date: Assumed `reference_date` (t0<t1)
            current_date: Assumed `current_date` (t1>t0)
            if_nan: What to do with `None`s (NaN). Could be a function or predefined states as follow:\n
                1. 'skip': do nothing (i.e. ignore `None`'s)
            type: `DOC_TYPE` used to use rules for matching tags and filling appropriately
        """
        return functional.aggregate_datetime(dataframe=self.dataframe, col_base_name=col_base_name,
                                             new_col_name=new_col_name, one_sided=one_sided,
                                             if_nan=if_nan, type=type,
                                             reference_date=reference_date,
                                             current_date=current_date)

    def file_specific_basic_transform(self, type: DOC_TYPES, path: str) -> pd.DataFrame:
        """
        Takes a specific file (see `DOC_TYPES`), then does data type fixing,
            missing value filling, descretization, etc.

        Remark: Since each files has its own unique tags and requirements,
            it is expected that all these transformation being hardcoded for each file,
            hence this method exists to just improve readability without any generalization
            to other problems or even files.

        args:
            type: The input document type (see `constant.DOC_TYPES`)  
        """

        raise NotImplementedError

    @loggingdecorator(logger.name+'.DataframePreprocessor.func', level=logging.DEBUG, output=False, input=True)
    def change_dtype(self, col_name: str, dtype: Callable,
                     if_nan: Union[str, Callable] = 'skip', **kwargs):
        """
        Takes a column name and changes the dataframe's column data type where for 
            None (nan) values behave based on `if_nan` argument.

        args:
            col_name: Column name of the dataframe
            dtype: target data type as a function e.g. `np.float32`
            if_nan: What to do with `None`s (NaN). Could be a function or predefined states as follow:\n
                1. 'skip': do nothing (i.e. ignore `None`'s)
                2. 
        """

        return functional.change_dtype(dataframe=self.dataframe, col_name=col_name,
                                       dtype=dtype, if_nan=if_nan, **kwargs)

    @loggingdecorator(logger.name+'.DataframePreprocessor.func', level=logging.DEBUG, output=False, input=True)
    def config_csv_to_dict(self, path: str) -> dict:
        """
        Take a config CSV and return a dictionary of key and values

        args:
            path: string path to config file
        """

        config_df = pd.read_csv(path)
        return dict(zip(config_df[config_df.columns[0]], config_df[config_df.columns[1]]))


class UnitConverter:
    """
    Contains utility tools for converting different units to each other.

    For including domain specific rules of conversion, extend this class for
        each category.e.g. for finance.
    """

    def __init__(self) -> None:
        pass

    def unit_converter(self, sparse: Union[float, None], dense: Union[float, None],
                       factor: float) -> float:
        """
        convert `sparse` or `dense` to each other using
            the rule of thump of `dense = (factor) sparse`.

        args:
            sparse: the smaller/sparser amount which is a percentage of `dense`,\n
                if provided calculates `sparse = (factor) dense`.
            dense: the larger/denser amount which is a multiplication of `sparse`,\n
                if provided calculates `dense = (factor) sparse`
            factor: sparse to dense factor, either directly provided as a\n
                float number or as a predefined factor given by `constant.FINANCIAL_RATIOS`
        """

        if sparse is not None:
            dense = factor * sparse
            return dense
        elif dense is not None:
            sparse = factor * dense
            return sparse
        else:
            raise ValueError('Only `sparse` or `dense` can be None.')


class FinancialUnitConverter(UnitConverter):
    """
    Contains utility tools for converting different financial units to each other.

    All functions that you implement should take the factor value using
        `self.CONSTANTS['function_name']`. E.g.::

        def rent2deposit(self, rent: float) -> float:
            self.unit_converter(sparse=rent, dense=None, factor=self.CONSTANTS['rent2deposit'])

    """

    def __init__(self, CONSTANTS: dict = FINANCIAL_RATIOS) -> None:
        """
        Gets constant values needed for conversion
        """
        super().__init__()
        self.CONSTANTS = CONSTANTS

    def rent2deposit(self, rent: float) -> float:
        return self.unit_converter(sparse=rent, dense=None,
                                   factor=self.CONSTANTS['rent2deposit'])

    def deposit2rent(self, deposit: float) -> float:
        return self.unit_converter(sparse=None, dense=deposit,
                                   factor=self.CONSTANTS['deposit2rent'])

    def deposit2worth(self, deposit: float) -> float:
        return self.unit_converter(sparse=deposit, dense=None,
                                   factor=self.CONSTANTS['deposit2worth'])

    def worth2deposit(self, worth: float) -> float:
        return self.unit_converter(sparse=None, dense=worth,
                                   factor=self.CONSTANTS['worth2deposit'])

    def tax2income(self, tax: float) -> float:
        return self.unit_converter(sparse=tax, dense=None,
                                   factor=self.CONSTANTS['tax2income'])

    def income2tax(self, income: float) -> float:
        return self.unit_converter(sparse=None, dense=income,
                                   factor=self.CONSTANTS['income2tax'])

    def income2worth(self, income: float) -> float:
        return self.unit_converter(sparse=income, dense=None,
                                   factor=self.CONSTANTS['income2worth'])

    def worth2income(self, worth: float) -> float:
        return self.unit_converter(sparse=None, dense=worth,
                                   factor=self.CONSTANTS['worth2income'])


class WorldBankXMLProcessor:
    """
    An XML processor which is customized ot handle data dumped from
        https://data.worldbank.org/indicator (since it is used by mainstream, works for us too.)

    It's recommended to extend this class to work with particular indicator by
        first filtering by a "indicator", then manipulating the resulting dataframe.

    *Remark:* we prefer querying over `Pandas` dataframe than `lxml`
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        args:
            dataframe: Main Pandas DataFrame to be processed
        """
        self.dataframe = dataframe

        # logging
        self.logger_name = '.WorldBankXMLProcessor'
        self.logger = logging.getLogger(logger.name+self.logger_name)

        # populate processed dict
        self.country_name_to_numeric_dict = self.indicator_filter()

    @loggingdecorator(logger.name+'.WorldBankXMLProcessor.func', level=logging.INFO,
                      output=False, input=True)
    def indicator_filter(self) -> dict:
        """
        Aggregates using mean operation over all columns except name/index. Also,
            values have to be scaled into [1, 7] range to match other
            world bank data processors.

        In this scenario, pivots a row-based dataframe to column based for 'Years' and 
            aggregates over them to achieve a 2-columns dataframe.

        """
        dataframe = self.dataframe
        # pivot XML attributes of `<field>` tag
        dataframe = dataframe.pivot(columns='name', values='field')
        dataframe = dataframe.drop('Item', axis=1)
        # fill None s created by pivoting (onehot to stacked) only over country names
        dataframe['Country or Area'] = dataframe['Country or Area'].ffill().bfill()
        dataframe = dataframe.drop_duplicates()  # drop repetition of onehots
        dataframe = dataframe.ffill().bfill()  # fill None of values of countries
        dataframe = self.__include_years(dataframe=dataframe)  # exclude old years
        # dataframe = dataframe[dataframe['Year'].astype(int) >= 2017]
        dataframe = dataframe.drop_duplicates(subset=['Country or Area', 'Year'],
                                              keep='last').reset_index()
        # pivot `Years` values as a set of separate columns i.e.
        #   from [name, years] -> [name, year1, year2, ...]
        df2 = dataframe.pivot(index='index', columns='Year', values='Value')
        # add names to pivoted years
        dataframe.drop('index', axis=1, inplace=True)
        dataframe.reset_index(inplace=True)
        df2.reset_index(inplace=True)
        dataframe = df2.join(dataframe['Country or Area'])
        # fill None s after pivoting `Years`
        country_names = dataframe['Country or Area'].unique()
        for cn in country_names:
            dataframe[dataframe['Country or Area'] ==
                      cn] = dataframe[dataframe['Country or Area'] == cn].ffill().bfill()
        # drop duplicates caused by filling None s of pivoting
        dataframe = dataframe.drop_duplicates(subset=['Country or Area'])

        # aggregation
        # drop scores/ranks and aggregate them into one column
        dataframe.drop('index', axis=1, inplace=True)
        mean_columns = [c for c in dataframe.columns.values if c.isnumeric()]
        dataframe['mean'] = dataframe[mean_columns].astype(float).mean(axis=1)
        dataframe.drop(dataframe.columns[:-2], axis=1, inplace=True)

        dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
            lambda x: x.lower())

        # scale to [1-7] (standard of World Data Bank)
        column_max = dataframe['mean'].max()
        column_min = dataframe['mean'].min()

        def standardize(x):
            return (((x - column_min) * (7. - 1.)) / (column_max - column_min)) + 1.
        dataframe['mean'] = dataframe['mean'].apply(standardize)
        return dict(zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]]))

    @staticmethod
    @loggingdecorator(logger.name+'.WorldBankXMLProcessor.func', level=logging.INFO,
                      output=False, input=False)
    def __include_years(dataframe: pd.DataFrame, start: Union[int, None] = None,
                        end: Union[int, None] = None) -> pd.DataFrame:
        """
        Processes a dataframe to only include years given tuple of years
            where `years=(start, end)`. Works inplace, hence manipulates original dataframe.

        *REMARK*: Currently only supports starting date. Defaults to `2017`. # TODO:
        args:
            dataframe: Pandas dataframe to be processed
            years: A tuple of `(start, end)` to limit years of data.
                If `None` (=default), all years will be included
        """
        start = 2017 if start is None else start

        assert end is None  # TODO: include end year
        dataframe = dataframe[dataframe['Year'].astype(int) >= start]
        return dataframe

    @loggingdecorator(logger.name+'.WorldBankXMLProcessor.func',
                      level=logging.DEBUG, output=True, input=True)
    def convert_country_name_to_numeric(self, string: str) -> float:
        """
        Converts the name of a country into a numerical value.
        """
        if string is None:
            string = 'Unknown'
        string = string.lower()
        # see `self.indicator_filter` for description of `1.` and `150` magic numbers
        return functional.search_dict(string=string,
                                      dic=self.country_name_to_numeric_dict, if_nan=1.)


class WorldBankDataframeProcessor:
    """
    A Pandas Dataframe processor which is customized to handle data dumped from 
        https://govdata360.worldbank.org (since it's used by mainstream, works for us too)

    It's recommended to extend this class to work with particular indicator by
        first filtering by a "indicator", then manipulating the resulting dataframe.
    """

    def __init__(self, dataframe: pd.DataFrame, subindicator_rank: bool = False) -> None:
        """
        Gets a raw dataframe, drops redundant columns, and prepares for extracting subsets
            given `include_years` and `filter_indicator`.

        args:
            dataframe: Main Pandas DataFrame to be processed
            subindicator_rank: Whether or not use ranking (discrete)
                or score (continuous) for given `indicator_name`. Defaults to `False`.
        """
        # set constants
        self.dataframe = dataframe
        self.INDICATOR = 'Indicator'
        self.SUBINDICATOR = 'Subindicator Type'
        self.subindicator_rank = subindicator_rank
        self.SUBINDICATOR_TYPE = 'Rank' if subindicator_rank else '1-7 Best'
        # drop useless columns
        columns_to_drop = ['Country ISO3', 'Indicator Id', ]
        columns_to_drop = columns_to_drop + \
            [c for c in dataframe.columns.values if '-' in c]
        self.dataframe.drop(columns_to_drop, axis=1, inplace=True)

        # logging
        self.logger_name = '.WorldBankDataframeProcessor'
        self.logger = logging.getLogger(logger.name+self.logger_name)

    @loggingdecorator(logger.name+'.WorldBankDataframeProcessor.func', level=logging.INFO,
                      output=True, input=True)
    def include_years(self, years: Tuple[Optional[int], Optional[int]] = None) -> None:
        """
        Processes a dataframe to only include years given tuple of `years`
            where `years=(start, end)`. Works inplace, hence manipulates original dataframe.

        args:
            years: A tuple of `(start, end)` to limit years of data.
                If `None` (=default), all years will be included
        """
        # figure out start and end year index of columns names values
        start_year, end_year = [
            str(y) for y in years] if years is not None else (None, None)
        column_years = [
            c for c in self.dataframe.columns.values if c.isnumeric()]
        start_year_index = column_years.index(
            start_year) if start_year is not None else 0
        end_year_index = column_years.index(
            end_year) if end_year is not None else -1
        # dataframe with desired years
        sub_column_years = column_years[start_year_index: end_year_index+1]
        columns_to_drop = [c for c in list(
            set(column_years) - set(sub_column_years)) if c.isnumeric()]
        self.dataframe.drop(columns_to_drop, axis=True, inplace=True)

    @loggingdecorator(logger.name+'.WorldBankDataframeProcessor.func', level=logging.INFO,
                      output=False, input=True)
    def indicator_filter(self, indicator_name: str) -> pd.DataFrame:
        """
        Filters the rows by given `indicator_name` and drops corresponding columns used
            for filtering. Then aggregates using mean operation.

        args:
            indicator_name: A string containing an indicator's full name
        """
        # filter rows that only contain the provided `indicator_name` with type `rank` or `score`
        dataframe = self.dataframe[(self.dataframe[self.INDICATOR] == indicator_name) &
                                   (self.dataframe[self.SUBINDICATOR] == self.SUBINDICATOR_TYPE)]
        dataframe.drop([self.INDICATOR, self.SUBINDICATOR],
                       axis=1, inplace=True)
        # drop scores/ranks and aggregate them into one column
        dataframe[indicator_name + '_mean'] = dataframe.mean(axis=1, skipna=True,
                                                             numeric_only=True)
        dataframe.drop(dataframe.columns[1:-1], axis=1, inplace=True)

        # add a default row when input country name is 'Unknown` (this value was hardcoded in XFA PDF LOV field)
        df_unknown = pd.DataFrame(
            {dataframe.columns[0]: ['Unknown'], dataframe.columns[1]: [None]})
        dataframe = pd.concat(objs=[dataframe, df_unknown], axis=0,
                              verify_integrity=True, ignore_index=True)

        # fillna since there is no info in the past years of that country -> unknown country
        if not self.subindicator_rank:  # fillna with lowest score = 1.
            dataframe = dataframe.fillna(value=1.)
        else:  # fillna with highest rank = 150
            dataframe = dataframe.fillna(value=150)
        return dataframe


class EducationCountryScoreDataframePreprocessor(WorldBankDataframeProcessor):
    """
    Handles `'Quality of the education system'` indicator of a `WorldBankDataframeProcessor`
        dataframe. The value ranges from 1 to 7 as score where higher is better.
    """

    def __init__(self, dataframe: pd.DataFrame, subindicator_rank: bool = False) -> None:
        super().__init__(dataframe, subindicator_rank)

        self.INDICATOR_NAME = 'Quality of the education system, 1-7 (best)'
        self.country_name_to_numeric_dict = self.__indicator_filter()

        # logging
        self.logger_name = '.WorldBankDataframeProcessor.EducationCountryScoreDataframePreprocessor'
        self.logger = logging.getLogger(logger.name+self.logger_name)

    @loggingdecorator(logger.name+'.WorldBankDataframeProcessor.EducationCountryScoreDataframePreprocessor.func',
                      level=logging.DEBUG, output=False, input=True)
    def __indicator_filter(self) -> dict:
        """
        Filters the rows by a constant `INDICATOR_NAME` defined by class type and
            drops corresponding columns used for filtering.
        """
        dataframe = self.indicator_filter(indicator_name=self.INDICATOR_NAME)
        dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
            lambda x: x.lower())
        return dict(zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]]))

    @loggingdecorator(logger.name+'.WorldBankDataframeProcessor.EducationCountryScoreDataframePreprocessor.func',
                      level=logging.DEBUG, output=True, input=True)
    def convert_country_name_to_numeric(self, string: str) -> float:
        """
        Converts the name of a country into a numerical value.

        args:
            string: country name in string. If `None`, will be filled with `'Unknown'`
        """
        if string is None:
            string = 'Unknown'
        string = string.lower()
        # see `self.indicator_filter` for description of `1.` and `150` magic numbers
        return functional.search_dict(string=string, dic=self.country_name_to_numeric_dict,
                                      if_nan=1. if not self.subindicator_rank else 150)


class EconomyCountryScoreDataframePreprocessor(WorldBankDataframeProcessor):
    """
    Handles `'Global Competitiveness Index'` indicator of
        a `WorldBankDataframeProcessor` dataframe. The value ranges from 1 to 7 as the score
        where higher is better.
    """

    def __init__(self, dataframe: pd.DataFrame, subindicator_rank: bool = False) -> None:
        super().__init__(dataframe, subindicator_rank)

        self.INDICATOR_NAME = 'Global Competitiveness Index'
        self.country_name_to_numeric_dict = self.__indicator_filter()

        # logging
        self.logger_name = '.WorldBankDataframeProcessor.EconomyCountryScoreDataframePreprocessor'
        self.logger = logging.getLogger(logger.name+self.logger_name)

    @loggingdecorator(logger.name+'.WorldBankDataframeProcessor.EconomyCountryScoreDataframePreprocessor.func',
                      level=logging.DEBUG, output=False, input=True)
    def __indicator_filter(self) -> dict:
        """
        Filters the rows by a constant `INDICATOR_NAME` defined by class type and
            drops corresponding columns used for filtering.
        """
        dataframe = self.indicator_filter(indicator_name=self.INDICATOR_NAME)
        dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
            lambda x: x.lower())
        return dict(zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]]))

    @loggingdecorator(logger.name+'.WorldBankDataframeProcessor.EconomyCountryScoreDataframePreprocessor.func',
                      level=logging.DEBUG, output=True, input=True)
    def convert_country_name_to_numeric(self, string: str) -> float:
        """
        Converts the name of a country into a numerical value.
        """
        if string is None:
            string = 'Unknown'
        string = string.lower()
        # see `self.indicator_filter` for description of `1.` and `150` magic numbers
        return functional.search_dict(string=string, dic=self.country_name_to_numeric_dict,
                                      if_nan=1. if not self.subindicator_rank else 150)


class CanadaDataframePreprocessor(DataframePreprocessor):
    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        super().__init__(dataframe)
        self.logger_name = '.CanadaDataframePreprocessor'
        self.logger = logging.getLogger(logger.name+self.logger_name)

        self.base_date = None  # the time forms were filled, considered "today" for forms

        # get country code to name dict
        self.config_path = CONFIGS_PATH.CANADA_COUNTRY_CODE_TO_NAME.value
        self.CANADA_COUNTRY_CODE_TO_NAME = self.config_csv_to_dict(
            self.config_path)

    @loggingdecorator(logger.name+'.CanadaDataframePreprocessor.func', level=logging.INFO, output=True, input=True)
    def convert_country_code_to_name(self, string: str) -> str:
        """
        Converts the (custom and non-standard) code of a country to its name given the XFA docs LOV section.
        # TODO: integrate this into `file_specific...` after verifying it in `'notebooks/data_exploration_dev.ipynb'`
        args:
            string: input code string
        """
        logger = logging.getLogger(
            self.logger.name+'.convert_country_code_to_name')

        country = [c for c in self.CANADA_COUNTRY_CODE_TO_NAME.keys()
                   if string in c]
        if country:
            return self.CANADA_COUNTRY_CODE_TO_NAME[country]
        else:
            logger.debug('"{}" country code could not be found in the config file="{}".'.format(
                string, self.config_path))
            return 'Unknown'  # '000' code in XFA forms

    @loggingdecorator(logger.name+'.CanadaDataframePreprocessor.func', level=logging.INFO, output=False, input=True)
    def file_specific_basic_transform(self, type: DOC_TYPES, path: str) -> pd.DataFrame:
        canada_xfa = CanadaXFA()  # Canada PDF to XML

        if type == DOC_TYPES.canada_5257e:
            # XFA to XML
            xml = canada_xfa.extract_raw_content(path)
            xml = canada_xfa.clean_xml_for_csv(
                xml=xml, type=DOC_TYPES.canada_5257e)
            # XML to flattened dict
            data_dict = canada_xfa.xml_to_flattened_dict(xml=xml)
            data_dict = canada_xfa.flatten_dict(data_dict)
            # clean flattened dict
            data_dict = functional.dict_summarizer(data_dict, cutoff_term=CANADA_CUTOFF_TERMS.ca5257e.value,
                                                   KEY_ABBREVIATION_DICT=CANADA_5257E_KEY_ABBREVIATION,
                                                   VALUE_ABBREVIATION_DICT=CANADA_5257E_VALUE_ABBREVIATION)
            # convert each data dict to a dataframe
            dataframe = pd.DataFrame.from_dict(
                data=[data_dict], orient='columns')
            self.dataframe = dataframe
            # drop pepeg columns
            #   warning: setting `errors='ignore` ignores errors if columns do not exist!
            dataframe.drop(CANADA_5257E_DROP_COLUMNS, axis=1,
                           inplace=True, errors='ignore')

            # Adult binary state: adult=True or child=False
            dataframe['P1.AdultFlag'] = dataframe['P1.AdultFlag'].apply(
                lambda x: True if x == 'adult' else False)
            # service language: 1=En, 2=Fr -> need to be changed to categorical
            dataframe = self.change_dtype(col_name='P1.PD.ServiceIn.ServiceIn', dtype=np.int8,
                                          if_nan='skip')
            # AliasNameIndicator: 1=True, 0=False
            dataframe['P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator'] = dataframe['P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator'].apply(
                lambda x: True if x == 'Y' else False)
            # VisaType: String -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.VisaType.VisaType', dtype=str,
                                          if_nan='fill', value='OTHER')
            # Birth City: String -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.PlaceBirthCity', dtype=str,
                                          if_nan='fill', value='OTHER')
            # Birth country: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.PlaceBirthCountry', dtype=str,
                                          if_nan='fill', value='IRAN')
            # citizen of: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.Citizenship.Citizenship', dtype=str,
                                          if_nan='fill', value='IRAN')
            # current country of residency: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.CurrCOR.Row2.Country', dtype=str,
                                          if_nan='fill', value='IRAN')
            # current country of residency status: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.CurrCOR.Row2.Status', dtype=np.int8,
                                          if_nan='fill', value=np.int8(6))  # 6=OTHER in the form
            # current country of residency other description: bool -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.CurrCOR.Row2.Other', dtype=bool,
                                          if_nan='fill', value=False)
            # validation date of information, i.e. current date: datetime
            dataframe = self.change_dtype(col_name='P3.Sign.C1CertificateIssueDate',
                                          dtype=parser.parse, if_nan='skip')
            # keep it so we can access for other file if that was None
            if not dataframe['P3.Sign.C1CertificateIssueDate'].isna().all():
                self.base_date = dataframe['P3.Sign.C1CertificateIssueDate']
            # date of birth in year: string -> datetime
            dataframe = self.change_dtype(col_name='P1.PD.DOBYear', dtype=parser.parse,
                                          if_nan='skip')
            # current country of residency period: None -> Datetime (=age period)
            dataframe = self.change_dtype(col_name='P1.PD.CurrCOR.Row2.FromDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P1.PD.DOBYear'])
            dataframe = self.change_dtype(col_name='P1.PD.CurrCOR.Row2.ToDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            # current country of residency period: Datetime -> int days
            dataframe = self.aggregate_datetime(col_base_name='P1.PD.CurrCOR.Row2',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None,
                                                current_date=None)
            # date of birth in year: datetime -> int days
            dataframe = self.aggregate_datetime(col_base_name='P1.PD.DOBYear',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=dataframe['P1.PD.DOBYear'],
                                                current_date=dataframe['P3.Sign.C1CertificateIssueDate'],
                                                one_sided='right')
            # delete tnx to P1.PD.CurrCOR.Row2
            self.column_dropper(string='P1.PD.CORDates', inplace=True)
            # has previous country of residency: bool -> categorical
            dataframe['P1.PD.PCRIndicator'] = dataframe['P1.PD.PCRIndicator'].apply(
                lambda x: True if x == 'Y' else False)

            # clean previous country of residency features
            country_tag_list = [
                c for c in dataframe.columns.values if 'P1.PD.PrevCOR.' in c]
            PREV_COUNTRY_MAX_FEATURES = 4
            for i in range(len(country_tag_list) // PREV_COUNTRY_MAX_FEATURES):
                # in XLA extracted file, this section start from `Row2` (ie. i+2)
                i += 2
                # previous country of residency 02: string -> categorical
                dataframe = self.change_dtype(col_name='P1.PD.PrevCOR.Row'+str(i)+'.Country', dtype=str,
                                              if_nan='fill', value='OTHER')
                # previous country of residency status 02: string -> categorical
                dataframe = self.change_dtype(col_name='P1.PD.PrevCOR.Row'+str(i)+'.Status',
                                              dtype=np.int8, if_nan='fill', value=np.int8(6))
                # previous country of residency 02 period (P1.PD.PrevCOR.Row2): string -> datetime -> int days
                dataframe = self.change_dtype(col_name='P1.PD.PrevCOR.Row'+str(i)+'.FromDate',
                                              dtype=parser.parse, if_nan='fill',
                                              value=dataframe['P3.Sign.C1CertificateIssueDate'])
                dataframe = self.change_dtype(col_name='P1.PD.PrevCOR.Row'+str(i)+'.ToDate',
                                              dtype=parser.parse, if_nan='fill',
                                              value=dataframe['P3.Sign.C1CertificateIssueDate'])
                dataframe = self.aggregate_datetime(col_base_name='P1.PD.PrevCOR.Row'+str(i),
                                                    type=DOC_TYPES.canada, new_col_name='Period',
                                                    reference_date=None,
                                                    current_date=None)
            # delete tnx to P1.PD.PrevCOR.Row2 and P1.PD.PrevCOR.Row3
            self.column_dropper(string='P1.PD.PCRDatesR', inplace=True)

            # apply from country of residency (cwa=country where apply): Y=True, N=False
            dataframe['P1.PD.SameAsCORIndicator'] = dataframe['P1.PD.SameAsCORIndicator'].apply(
                lambda x: True if x == 'Y' else False)
            # country where applying: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.CWA.Row2.Country', dtype=str,
                                          if_nan='fill', value='OTHER')
            # country where applying status: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.CWA.Row2.Status',
                                          dtype=np.int8, if_nan='fill', value=np.int8(6))
            # country where applying other: string -> categorical
            dataframe = self.change_dtype(col_name='P1.PD.CWA.Row2.Other', dtype=bool,
                                          if_nan='fill', value=False)
            # country where applying period: datetime -> int days
            dataframe = self.change_dtype(col_name='P1.PD.CWA.Row2.FromDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.change_dtype(col_name='P1.PD.CWA.Row2.ToDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            # TODO: if None (here .Period=0): fill with average statistically
            dataframe = self.aggregate_datetime(col_base_name='P1.PD.CWA.Row2',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None,
                                                current_date=None)
            # delete tnx to P1.PD.CWA.Row2
            self.column_dropper(string='P1.PD.CWADates', inplace=True)
            # marriage period: datetime -> int days
            dataframe = self.change_dtype(col_name='P1.MS.SecA.DateOfMarr',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.aggregate_datetime(col_base_name='P1.MS.SecA.DateOfMarr',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None,
                                                current_date=dataframe['P3.Sign.C1CertificateIssueDate'],
                                                one_sided='right')
            # delete tnx to P1.MS.SecA.DateOfMarr
            self.column_dropper(string='P1.MS.SecA.MarrDate.From')
            # previous marriage: Y=True, N=False
            dataframe['P2.MS.SecA.PrevMarrIndicator'] = dataframe['P2.MS.SecA.PrevMarrIndicator'].apply(
                lambda x: True if x == 'Y' else False)
            # previous marriage type of relationship
            dataframe = self.change_dtype(col_name='P2.MS.SecA.TypeOfRelationship',
                                          dtype=str, if_nan='fill',
                                          value='OTHER')
            # previous spouse age period: string -> datetime -> int days
            dataframe = self.change_dtype(col_name='P2.MS.SecA.PrevSpouseDOB.DOBYear',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.aggregate_datetime(col_base_name='P2.MS.SecA.PrevSpouseDOB.DOBYear',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None, one_sided='right',
                                                current_date=dataframe['P3.Sign.C1CertificateIssueDate'])
            # previous marriage period: string -> datetime -> int days
            dataframe = self.change_dtype(col_name='P2.MS.SecA.FromDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.change_dtype(col_name='P2.MS.SecA.ToDate.ToDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.aggregate_datetime(col_base_name='P2.MS.SecA',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None, current_date=None)
            self.column_dropper(string='P2.MS.SecA.Prevly', inplace=True)
            # passport country of issue: string -> categorical
            dataframe = self.change_dtype(col_name='P2.MS.SecA.Psprt.CountryofIssue.CountryofIssue',
                                          dtype=str, if_nan='fill', value='OTHER')
            # expiry remaining period: datetime -> int days
            # if None, fill with 1 year ago, ie. period=1year
            temp_date = dataframe['P3.Sign.C1CertificateIssueDate'].apply(
                lambda x: x+relativedelta(years=-1))
            dataframe = self.change_dtype(col_name='P2.MS.SecA.Psprt.ExpiryDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=temp_date)
            dataframe = self.aggregate_datetime(col_base_name='P2.MS.SecA.Psprt.ExpiryDate',
                                                type=DOC_TYPES.canada, new_col_name='Remaining',
                                                current_date=None, one_sided='left',
                                                reference_date=dataframe['P3.Sign.C1CertificateIssueDate'])
            # native lang: string -> categorical
            dataframe = self.change_dtype(col_name='P2.MS.SecA.Langs.languages.nativeLang.nativeLang',
                                          dtype=str, if_nan='fill', value='IRAN')
            # communication lang: Eng, Fr, both, none -> categorical
            dataframe = self.change_dtype(col_name='P2.MS.SecA.Langs.languages.ableToCommunicate.ableToCommunicate',
                                          dtype=str, if_nan='fill', value='NEITHER')
            # language official test: bool -> binary
            dataframe['P2.MS.SecA.Langs.LangTest'] = dataframe['P2.MS.SecA.Langs.LangTest'].apply(
                lambda x: True if x == 'Y' else False)
            # have national ID: bool -> binary
            dataframe['P2.natID.q1.natIDIndicator'] = dataframe['P2.natID.q1.natIDIndicator'].apply(
                lambda x: True if x == 'Y' else False)
            # national ID country of issue: string -> categorical
            dataframe = self.change_dtype(col_name='P2.natID.natIDdocs.CountryofIssue.CountryofIssue',
                                          dtype=str, if_nan='fill', value='IRAN')
            # United States doc: bool -> binary
            dataframe['P2.USCard.q1.usCardIndicator'] = dataframe['P2.USCard.q1.usCardIndicator'].apply(
                lambda x: True if x == 'Y' else False)
            # drop contact information except having US Canada phone number
            self.column_dropper(string='P2.CI.cntct',
                                exclude='CanadaUS', inplace=True)
            # US Canada phone number: bool -> binary
            dataframe['P2.CI.cntct.PhnNums.Phn.CanadaUS'] = dataframe['P2.CI.cntct.PhnNums.Phn.CanadaUS'].apply(
                lambda x: True if x == '1' else False)
            # US Canada alt phone number: bool -> binary
            dataframe['P2.CI.cntct.PhnNums.AltPhn.CanadaUS'] = dataframe['P2.CI.cntct.PhnNums.AltPhn.CanadaUS'].apply(
                lambda x: True if x == '1' else False)
            # purpose of visit: string, 8 states -> categorical
            dataframe = self.change_dtype(col_name='P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit',
                                          dtype=np.int8, if_nan='fill',
                                          value=np.int8(7))  # 7 is other in the form
            # purpose of visit description: string -> binary
            dataframe = self.change_dtype(col_name='P3.DOV.PrpsRow1.Other.Other', dtype=bool,
                                          if_nan='fill', value=False)
            # how long going to stay: None -> datetime (0 days)
            dataframe = self.change_dtype(col_name='P3.DOV.PrpsRow1.HLS.FromDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.change_dtype(col_name='P3.DOV.PrpsRow1.HLS.ToDate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            # how long going to stay: datetime -> int days
            dataframe = self.aggregate_datetime(col_base_name='P3.DOV.PrpsRow1.HLS',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None, current_date=None)
            # delete tnx to P3.DOV.PrpsRow1.HLS
            self.column_dropper(string='P3.DOV.PrpsRow1.HLS',
                                exclude='Period', inplace=True)
            # fund to integer
            dataframe = self.change_dtype(col_name='P3.DOV.PrpsRow1.Funds.Funds', dtype=np.int32,
                                          if_nan='skip')
            # relation to applicant of purpose of visit 01: string -> categorical
            dataframe = self.change_dtype(col_name='P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe',
                                          dtype=str, if_nan='fill', value='OTHER')
            # relation to applicant of purpose of visit 02: string -> categorical
            dataframe = self.change_dtype(col_name='P3.cntcts_Row2.Relationship.RelationshipToMe',
                                          dtype=str, if_nan='fill', value='OTHER')
            # higher education: bool -> binary
            dataframe['P3.Edu.EduIndicator'] = dataframe['P3.Edu.EduIndicator'].apply(
                lambda x: True if x == 'Y' else False)
            # higher education period: string -> datetime -> int days
            dataframe = self.change_dtype(col_name='P3.Edu.Edu_Row1.FromYear',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.change_dtype(col_name='P3.Edu.Edu_Row1.ToYear',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['P3.Sign.C1CertificateIssueDate'])
            dataframe = self.aggregate_datetime(col_base_name='P3.Edu.Edu_Row1',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None, current_date=None)
            # higher education country: string -> categorical
            # TODO: skip here, and only fill with 'IRAN' for those who have 'P3.Edu.EduIndicator' = True
            #   see ### P3.Edu.Edu_Row1.Country.Country -> categorical in notebooks for more info
            dataframe = self.change_dtype(col_name='P3.Edu.Edu_Row1.Country.Country',
                                          dtype=str, if_nan='fill', value='IRAN')
            # TODO: see #1
            # field of study: string -> categorical
            dataframe['P3.Edu.Edu_Row1.FieldOfStudy'] = dataframe['P3.Edu.Edu_Row1.FieldOfStudy'].astype('string')
            # clean occupation features
            occupation_tag_list = [
                c for c in dataframe.columns.values if 'P3.Occ.OccRow' in c]
            PREV_OCCUPATION_MAX_FEATURES = 4
            for i in range(len(occupation_tag_list) // PREV_OCCUPATION_MAX_FEATURES):
                i += 1  # in the form, it starts from Row1 (ie. i+1)
                # occupation period 01: none -> string year -> int days
                dataframe = self.change_dtype(col_name='P3.Occ.OccRow'+str(i)+'.FromYear',
                                              dtype=parser.parse, if_nan='fill',
                                              value=dataframe['P3.Sign.C1CertificateIssueDate'])
                dataframe = self.change_dtype(col_name='P3.Occ.OccRow'+str(i)+'.ToYear',
                                              dtype=parser.parse, if_nan='fill',
                                              value=dataframe['P3.Sign.C1CertificateIssueDate'])
                dataframe = self.aggregate_datetime(col_base_name='P3.Occ.OccRow'+str(i),
                                                    type=DOC_TYPES.canada, new_col_name='Period',
                                                    reference_date=None, current_date=None)
                # TODO: see #2
                # occupation type 01: string -> categorical
                dataframe = self.change_dtype(col_name='P3.Occ.OccRow' + str(i)+'.Occ.Occ', dtype=str,
                                              if_nan='fill', value='OTHER')
                # occupation country: string -> categorical
                dataframe = self.change_dtype(col_name='P3.Occ.OccRow' + str(i)+'.Country.Country',
                                              dtype=str, if_nan='fill', value='IRAN')

            # medical details: string -> binary
            dataframe = self.change_dtype(col_name='P3.BGI.Details.MedicalDetails', dtype=bool,
                                          if_nan='fill', value=False)
            # other than medical: string -> binary
            dataframe = self.change_dtype(col_name='P3.BGI.otherThanMedic', dtype=bool,
                                          if_nan='fill', value=False)
            # without authentication stay, work, etc: bool -> binary
            dataframe['P3.noAuthStay'] = dataframe['P3.noAuthStay'].apply(
                lambda x: True if x == 'Y' else False)
            # deported or refused entry: bool -> binary
            dataframe['P3.refuseDeport'] = dataframe['P3.refuseDeport'].apply(
                lambda x: True if x == 'Y' else False)
            # previously applied: bool -> binary
            dataframe['P3.BGI2.PrevApply'] = dataframe['P3.BGI2.PrevApply'].apply(
                lambda x: True if x == 'Y' else False)
            # criminal record: bool -> binary
            dataframe['P3.PWrapper.criminalRec'] = dataframe['P3.PWrapper.criminalRec'].apply(
                lambda x: True if x == 'Y' else False)
            # military record: bool -> binary
            dataframe['P3.PWrapper.Military.Choice'] = dataframe['P3.PWrapper.Military.Choice'].apply(
                lambda x: True if x == 'Y' else False)
            # political, violent movement record: bool -> binary
            dataframe['P3.PWrapper.politicViol'] = dataframe['P3.PWrapper.politicViol'].apply(
                lambda x: True if x == 'Y' else False)
            # witness of ill treatment: bool -> binary
            dataframe['P3.PWrapper.witnessIllTreat'] = dataframe['P3.PWrapper.witnessIllTreat'].apply(
                lambda x: True if x == 'Y' else False)
            # drop the time form was filled
            self.column_dropper(
                string='P3.Sign.C1CertificateIssueDate', inplace=True)

            return dataframe

        if type == DOC_TYPES.canada_5645e:
            # XFA to XML
            xml = canada_xfa.extract_raw_content(path)
            xml = canada_xfa.clean_xml_for_csv(
                xml=xml, type=DOC_TYPES.canada_5645e)
            # XML to flattened dict
            data_dict = canada_xfa.xml_to_flattened_dict(xml=xml)
            data_dict = canada_xfa.flatten_dict(data_dict)
            # clean flattened dict
            data_dict = functional.dict_summarizer(data_dict, cutoff_term=CANADA_CUTOFF_TERMS.ca5645e.value,
                                                   KEY_ABBREVIATION_DICT=CANADA_5645E_KEY_ABBREVIATION,
                                                   VALUE_ABBREVIATION_DICT=None)

            # convert each data dict to a dataframe
            dataframe = pd.DataFrame.from_dict(
                data=[data_dict], orient='columns')
            self.dataframe = dataframe

            # drop pepeg columns
            #   warning: setting `errors='ignore` ignores errors if columns do not exist!
            dataframe.drop(CANADA_5645E_DROP_COLUMNS, axis=1,
                           inplace=True, errors='ignore')

            # transform multiple pleb columns into a single chad one and fixing column dtypes
            # type of application: (already onehot) string -> int
            cols = [col for col in dataframe.columns.values if 'p1.Subform1' in col]
            for c in cols:
                dataframe = self.change_dtype(col_name=c,
                                              dtype=np.int16, if_nan='fill',
                                              value=np.int16('0'))
            # drop all names
            self.column_dropper(string='Name', inplace=True)
            # drop all addresses
            self.column_dropper(string='Addr', inplace=True)
            # drop all Accompany=No and only rely on Accompany=Yes using binary state
            self.column_dropper(string='No', inplace=True)
            # applicant marriage status: string to integer
            dataframe = self.change_dtype(col_name='p1.SecA.App.ChdMStatus',
                                          dtype=np.int16, if_nan='fill',
                                          value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value))
            # validation date of information, i.e. current date: datetime
            dataframe = self.change_dtype(col_name='p1.SecC.SecCdate',
                                          dtype=parser.parse, if_nan='fill',
                                          value=self.base_date)
            # spouse date of birth: string -> datetime
            dataframe = self.change_dtype(col_name='p1.SecA.Sps.SpsDOB',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['p1.SecC.SecCdate'])
            # spouse age period: datetime -> int days
            dataframe = self.aggregate_datetime(col_base_name='p1.SecA.Sps.SpsDOB',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None,
                                                current_date=dataframe['p1.SecC.SecCdate'],
                                                one_sided='right')
            # spouse country of birth: string -> categorical
            dataframe = self.change_dtype(col_name='p1.SecA.Sps.SpsCOB',
                                          dtype=str, if_nan='skip')
            # spouse occupation type (issue #2): string -> categorical
            dataframe = self.change_dtype(col_name='p1.SecA.Sps.SpsOcc',
                                          dtype=str, if_nan='fill', value='OTHER')
            # spouse accompanying: coming=True or not_coming=False
            dataframe['p1.SecA.Sps.SpsAccomp'] = dataframe['p1.SecA.Sps.SpsAccomp'].apply(
                lambda x: True if x == '1' else False)
            # mother date of birth: string -> datetime
            dataframe = self.change_dtype(col_name='p1.SecA.Mo.MoDOB',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['p1.SecC.SecCdate'])
            # mother age period: datetime -> int days
            # TODO: if =0 (originally None), fill with average age difference of parents and children
            dataframe = self.aggregate_datetime(col_base_name='p1.SecA.Mo.MoDOB',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None,
                                                current_date=dataframe['p1.SecC.SecCdate'],
                                                one_sided='right')
            # mother occupation type (issue #2): string -> categorical
            dataframe = self.change_dtype(col_name='p1.SecA.Mo.MoOcc',
                                          dtype=str, if_nan='fill', value='OTHER')
            # mother marriage status: int -> categorical
            dataframe = self.change_dtype(col_name='p1.SecA.Mo.ChdMStatus',
                                          dtype=np.int16, if_nan='fill',
                                          value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value))
            # mother accompanying: coming=True or not_coming=False
            dataframe['p1.SecA.Mo.MoAccomp'] = dataframe['p1.SecA.Mo.MoAccomp'].apply(
                lambda x: True if x == '1' else False)
            # father date of birth: string -> datetime
            dataframe = self.change_dtype(col_name='p1.SecA.Fa.FaDOB',
                                          dtype=parser.parse, if_nan='fill',
                                          value=dataframe['p1.SecC.SecCdate'])
            # father age period: datetime -> int days
            # TODO: if =0 (originally None), fill with average age difference of parents and children
            dataframe = self.aggregate_datetime(col_base_name='p1.SecA.Fa.FaDOB',
                                                type=DOC_TYPES.canada, new_col_name='Period',
                                                reference_date=None,
                                                current_date=dataframe['p1.SecC.SecCdate'],
                                                one_sided='right')
            # mother occupation type (issue #2): string -> categorical
            dataframe = self.change_dtype(col_name='p1.SecA.Fa.FaOcc',
                                          dtype=str, if_nan='fill', value='OTHER')
            # father marriage status: int -> categorical
            dataframe = self.change_dtype(col_name='p1.SecA.Fa.ChdMStatus',
                                          dtype=np.int16, if_nan='fill',
                                          value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value))
            # father accompanying: coming=True or not_coming=False
            dataframe['p1.SecA.Fa.FaAccomp'] = dataframe['p1.SecA.Fa.FaAccomp'].apply(
                lambda x: True if x == '1' else False)

            # children's status
            children_tag_list = [
                c for c in dataframe.columns.values if 'p1.SecB.Chd' in c]
            CHILDREN_MAX_FEATURES = 6
            for i in range(len(children_tag_list) // CHILDREN_MAX_FEATURES):
                # child's marriage status 01: string to integer
                dataframe = self.change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdMStatus',
                                              dtype=np.int16, if_nan='fill',
                                              value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value))
                # child's relationship 01: string -> categorical
                dataframe = self.change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdRel',
                                              dtype=str, if_nan='fill', value='OTHER')
                # child's date of birth 01: string -> datetime
                dataframe = self.change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdDOB',
                                              dtype=parser.parse, if_nan='skip')
                # child's age period 01: datetime -> int days
                dataframe = self.aggregate_datetime(type=DOC_TYPES.canada,
                                                    col_base_name='p1.SecB.Chd.[' +
                                                    str(i)+'].ChdDOB', new_col_name='Period',
                                                    reference_date=None, one_sided='right',
                                                    current_date=dataframe['p1.SecC.SecCdate'],
                                                    if_nan='skip')
                # child's country of birth 01: string -> categorical
                dataframe = self.change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdCOB',
                                              dtype=str, if_nan='fill', value='IRAN')
                # child's occupation type 01 (issue #2): string -> categorical
                dataframe = self.change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdOcc',
                                              dtype=str, if_nan='fill', value='OTHER')
                # child's marriage status: int -> categorical
                dataframe = self.change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdMStatus',
                                              dtype=np.int16, if_nan='fill',
                                              value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value))
                # child's accompanying 01: coming=True or not_coming=False
                dataframe['p1.SecB.Chd.['+str(i)+'].ChdAccomp'] = dataframe['p1.SecB.Chd.['+str(i)+'].ChdAccomp'].apply(
                    lambda x: True if x == '1' else False)

                # check if the child does not exist and fill it properly (ghost case monkaS)
                if (dataframe['p1.SecB.Chd.['+str(i)+'].ChdMStatus'] == CANADA_FILLNA.ChdMStatus_5645e.value).all() \
                        and (dataframe['p1.SecB.Chd.['+str(i)+'].ChdRel'] == 'OTHER').all() \
                        and (dataframe['p1.SecB.Chd.['+str(i)+'].ChdDOB'].isna()).all() \
                        and (dataframe['p1.SecB.Chd.['+str(i)+'].ChdAccomp'] == False).all():
                    # ghost child's date of birth: None -> datetime (current date) -> 0 days
                    dataframe = self. change_dtype(col_name='p1.SecB.Chd.['+str(i)+'].ChdDOB',
                                                   dtype=parser.parse, if_nan='fill',
                                                   value=dataframe['p1.SecC.SecCdate'])
                    # ghost child's age period: datetime (current date) -> int 0 days
                    dataframe = self.aggregate_datetime(type=DOC_TYPES.canada,
                                                        col_base_name='p1.SecB.Chd.[' +
                                                        str(i)+'].ChdDOB', new_col_name='Period',
                                                        reference_date=None, one_sided='right',
                                                        current_date=dataframe['p1.SecC.SecCdate'],
                                                        if_nan=None)

            # fill existing child's date of birth where it is None with a heuristic
            # take average age period of children
            col_names = []  # holds all age periods
            col_names_age_all = []  # holds all age periods and date of births
            for i in range(len(children_tag_list) // CHILDREN_MAX_FEATURES):
                col_name = 'p1.SecB.Chd.['+str(i)+'].ChdDOB.Period'
                if col_name in dataframe.columns.values:
                    col_names.append(col_name)
                col_name = 'p1.SecB.Chd.['+str(i)+'].ChdDOB'
                if col_name in dataframe.columns.values:
                    col_names_age_all.append(col_name)
            # extract `Chd.DOB` from `Chd.DOB.Period`
            col_names_unprocessed = list(
                set(col_names_age_all) - set(col_names))
            for c in col_names_unprocessed:  # drop columns after processing them
                # average of family children as the heuristic
                dataframe[c+'.Period'] = dataframe[dataframe[col_names]
                                                   != 0].mean(axis=1, numeric_only=True)
                dataframe.drop(c, axis=1, inplace=True)

            # siblings' status
            siblings_tag_list = [
                c for c in dataframe.columns.values if 'p1.SecC.Chd' in c]
            SIBLINGS_MAX_FEATURES = 6
            for i in range(len(siblings_tag_list) // CHILDREN_MAX_FEATURES):
                # sibling's marriage status 01: string to integer
                dataframe = self.change_dtype(col_name='p1.SecC.Chd.['+str(i)+'].ChdMStatus',
                                              dtype=np.int16, if_nan='fill',
                                              value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value))
                # sibling's relationship 01: string -> categorical
                dataframe = self.change_dtype(col_name='p1.SecC.Chd.['+str(i)+'].ChdRel',
                                              dtype=str, if_nan='fill', value='OTHER')
                # sibling's date of birth 01: string -> datetime
                dataframe = self.change_dtype(col_name='p1.SecC.Chd.['+str(i)+'].ChdDOB',
                                              dtype=parser.parse, if_nan='skip')
                # sibling's age period 01: datetime -> int days
                dataframe = self.aggregate_datetime(type=DOC_TYPES.canada,
                                                    col_base_name='p1.SecC.Chd.[' +
                                                    str(i)+'].ChdDOB', new_col_name='Period',
                                                    reference_date=None, one_sided='right',
                                                    current_date=dataframe['p1.SecC.SecCdate'],
                                                    if_nan='skip')
                # sibling's country of birth 01: string -> categorical
                dataframe = self.change_dtype(col_name='p1.SecC.Chd.['+str(i)+'].ChdCOB',
                                              dtype=str, if_nan='fill', value='IRAN')
                # sibling's occupation type 01 (issue #2): string -> categorical
                dataframe = self.change_dtype(col_name='p1.SecC.Chd.['+str(i)+'].ChdOcc',
                                              dtype=str, if_nan='fill', value='OTHER')
                # sibling's accompanying: coming=True or not_coming=False
                dataframe['p1.SecC.Chd.['+str(i)+'].ChdAccomp'] = dataframe['p1.SecC.Chd.['+str(i)+'].ChdAccomp'].apply(
                    lambda x: True if x == '1' else False)

                # check if the sibling does not exist and fill it properly (ghost case monkaS)
                if (dataframe['p1.SecC.Chd.['+str(i)+'].ChdMStatus'] == CANADA_FILLNA.ChdMStatus_5645e.value).all() \
                        and (dataframe['p1.SecC.Chd.['+str(i)+'].ChdRel'] == 'OTHER').all() \
                        and (dataframe['p1.SecC.Chd.['+str(i)+'].ChdOcc'].isna()).all() \
                        and (dataframe['p1.SecC.Chd.['+str(i)+'].ChdAccomp'] == False).all():
                    # ghost sibling's date of birth: None -> datetime (current date) -> 0 days
                    dataframe = self.change_dtype(col_name='p1.SecC.Chd.['+str(i)+'].ChdDOB',
                                                  dtype=parser.parse, if_nan='fill',
                                                  value=dataframe['p1.SecC.SecCdate'])
                    # ghost sibling's age period: datetime (current date) -> int 0 days
                    dataframe = self.aggregate_datetime(type=DOC_TYPES.canada,
                                                        col_base_name='p1.SecC.Chd.[' +
                                                        str(i)+'].ChdDOB', new_col_name='Period',
                                                        reference_date=None, one_sided='right',
                                                        current_date=dataframe['p1.SecC.SecCdate'],
                                                        if_nan=None)

            # fill existing sibling's date of birth where it is None with a heuristic
            # take average age period of siblings
            col_names = []  # holds all age periods
            col_names_age_all = []  # holds all age periods and date of births
            for i in range(len(siblings_tag_list) // SIBLINGS_MAX_FEATURES):
                col_name = 'p1.SecC.Chd.['+str(i)+'].ChdDOB.Period'
                if col_name in dataframe.columns.values:
                    col_names.append(col_name)
                col_name = 'p1.SecC.Chd.['+str(i)+'].ChdDOB'
                if col_name in dataframe.columns.values:
                    col_names_age_all.append(col_name)
            # extract `Chd.DOB` from `Chd.DOB.Period`
            col_names_unprocessed = list(
                set(col_names_age_all) - set(col_names))
            for c in col_names_unprocessed:  # drop columns after processing them
                # average of family siblings as the heuristic
                dataframe[c+'.Period'] = dataframe[dataframe[col_names]
                                                   != 0].mean(axis=1, numeric_only=True)
                dataframe.drop(c, axis=1, inplace=True)

            # drop the time form was filled
            self.column_dropper(string='p1.SecC.SecCdate', inplace=True)

            return dataframe

        if type == DOC_TYPES.canada_label:
            dataframe = pd.read_csv(path, sep=' ', names=['VisaResult'])
            functional.change_dtype(dataframe=dataframe, col_name='VisaResult', dtype=np.int8,
                                    if_nan='fill', value=np.int8(0))
            return dataframe


class FileTransform:
    """
    A base class for applying transforms as a composable object over files. Any behavior
        over the files itself (not the content by any means) must extend this class.

    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(logger.name+'.FileTransform')
        pass

    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        """

        Args:
            src: source file to be processed
            dst: the pass that the processed file to be saved 
        """
        pass


class CopyFile(FileTransform):
    """
    Only copies a file, a wrapper around `shutil`'s copying methods. 
    Default is set to 'cf', i.e. `shutil.copyfile`
    """

    def __init__(self, mode: str) -> None:
        super().__init__()

        # see https://stackoverflow.com/a/30359308/18971263
        self.COPY_MODES = ['c', 'cf', 'c2']
        self.mode = mode if mode is not None else 'cf'
        self.__check_mode(mode=mode)

    @loggingdecorator(logger.name+'.FileTransform.CopyFile', level=logging.DEBUG,
                      input=True, output=False)
    def __call__(self, src: str, dst: str,  *args: Any, **kwds: Any) -> Any:
        if self.mode == 'c':
            shutil.copy(src=src, dst=dst)
        elif self.mode == 'cf':
            shutil.copyfile(src=src, dst=dst)
        elif self.mode == 'c2':
            shutil.copy2(src=src, dst=dst)

    def __check_mode(self, mode: str):
        """
        Checks copying mode to be in `shutil`
        """
        if not mode in self.COPY_MODES:
            raise ValueError(
                'Mode {} does not exist, choose one of "{}".'.format(mode, self.COPY_MODES))


class MakeContentCopyProtectedMachineReadable(FileTransform):
    """
    reads a 'content-copy' protected PDF and removes this restriction
        by saving a "printed" version of.

    Ref: https://www.reddit.com/r/Python/comments/t32z2o/simple_code_to_unlock_all_readonly_pdfs_in/

    args:
        file_name: file name, if None, considers all files in `src_path`
        src: source file path
        dst: destination (processed) file path 
    """

    def __init__(self) -> None:
        super().__init__()

    @loggingdecorator(logger.name+'.FileTransform.MakeContentCopyProtectMachineReadable',
                      level=logging.DEBUG, input=True, output=False)
    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        pdf = pikepdf.open(src, allow_overwriting_input=True)
        pdf.save(dst)


class FileTransformCompose:
    """
    Composes several transforms operating on files together, only applying functions
        on files that match the keyword using a dictionary

    """

    def __init__(self, transforms: dict) -> None:
        """
        Transformation dictionary over files in the following structure::
        {FileTransform: 'filter_str', ...}
        """
        if transforms is not None:
            for k in transforms.keys():
                if not issubclass(k.__class__, FileTransform):
                    raise TypeError(
                        'Keys must be {} instance.'.format(FileTransform))

        self.transforms = transforms

    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        for transform, file_filter in self.transforms.items():
            if file_filter in src:
                transform(src, dst)
