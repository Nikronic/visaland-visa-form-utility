__all__ = [
    "DataframePreprocessor",
    "CanadaDataframePreprocessor",
    "UnitConverter",
    "FinancialUnitConverter",
    "T0",
    "FileTransformCompose",
    "FileTransform",
    "CopyFile",
    "MakeContentCopyProtectedMachineReadable",
    "EducationCountryScoreDataframePreprocessor",
    "EconomyCountryScoreDataframePreprocessor",
    "WorldBankXMLProcessor",
    "WorldBankDataframeProcessor",
]

import logging
import shutil
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import pikepdf
from dateutil import parser
from dateutil.relativedelta import *

from vizard.configs import CANADA_COUNTRY_CODE_TO_NAME
from vizard.data import functional
from vizard.data.constant import *
from vizard.data.pdf import CanadaXFA

# logging
logger = logging.getLogger(__name__)

T0 = "19000202T000000"  # a default meaningless time to fill the `None`s


class DataframePreprocessor:
    """A wrapper around builtin Pandas functions to make it easier for our data values

    A class that contains methods for dealing with dataframes regarding
    transformation of data such as filling missing values, dropping columns,
    or aggregating multiple columns into a single more meaningful one.

    This class needs to be extended for file specific preprocessing where tags are
    unique and need to be done entirely manually.
    In this case, :func:`file_specific_basic_transform` needs to be implemented.
    """

    def __init__(self, dataframe: Optional[pd.DataFrame] = None) -> None:
        """

        Args:
            dataframe (Optional[:class:`pandas.DataFrame`], optional): Main dataframe to be preprocessed.
                Defaults to None.
        """
        self.dataframe = dataframe
        self.logger = logging.getLogger(logger.name + ".DataframePreprocessor")

    def column_dropper(
        self,
        string: str,
        exclude: str = None,
        regex: bool = False,
        inplace: bool = True,
    ) -> Optional[pd.DataFrame]:
        """See :func:`vizard.data.functional.column_dropper` for more information"""

        return functional.column_dropper(
            dataframe=self.dataframe,
            string=string,
            exclude=exclude,
            regex=regex,
            inplace=inplace,
        )

    def fillna_datetime(
        self,
        col_base_name: str,
        type: DOC_TYPES,
        one_sided: str | bool,
        date: Optional[str] = None,
        inplace: bool = False,
    ) -> Optional[pd.DataFrame]:
        """See :func:`vizard.data.functional.fillna_datetime` for more details"""
        if date is None:
            date = T0

        return functional.fillna_datetime(
            dataframe=self.dataframe,
            col_base_name=col_base_name,
            one_sided=one_sided,
            date=date,
            inplace=inplace,
            type=type,
        )

    def aggregate_datetime(
        self,
        col_base_name: str,
        new_col_name: str,
        type: DOC_TYPES,
        if_nan: Optional[str | Callable] = None,
        one_sided: Optional[str] = None,
        reference_date: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """See :func:`vizard.data.functional.aggregate_datetime` for more details"""
        return functional.aggregate_datetime(
            dataframe=self.dataframe,
            col_base_name=col_base_name,
            new_col_name=new_col_name,
            one_sided=one_sided,
            if_nan=if_nan,
            type=type,
            reference_date=reference_date,
            current_date=current_date,
        )

    def file_specific_basic_transform(self, type: DOC_TYPES, path: str) -> pd.DataFrame:
        """Takes a specific file then does data type fixing, missing value filling, discretization, etc.

        Note:
            Since each files has its own unique tags and requirements,
            it is expected that all these transformation being hardcoded for each file,
            hence this method exists to just improve readability without any generalization
            to other problems or even files.

        Args:
            type (DOC_TYPES): The input document type
                (see :class:`DOC_TYPES <vizard.data.constant.DOC_TYPES>`)
            path (str): Path to the file
        """

        raise NotImplementedError

    def change_dtype(
        self,
        col_name: str,
        dtype: Callable,
        if_nan: Optional[str | Callable] = "skip",
        **kwargs,
    ):
        """See :func:`vizard.data.functional.change_dtype` for more details"""

        return functional.change_dtype(
            dataframe=self.dataframe,
            col_name=col_name,
            dtype=dtype,
            if_nan=if_nan,
            **kwargs,
        )

    def config_csv_to_dict(self, path: str) -> dict:
        """
        Take a config CSV and return a dictionary of key and values

        args:
            path: string path to config file
        """

        config_df = pd.read_csv(path)
        return dict(
            zip(config_df[config_df.columns[0]], config_df[config_df.columns[1]])
        )


class UnitConverter:
    """
    Contains utility tools for converting different units to each other.

    For including domain specific rules of conversion, extend this class for
    each category, e.g. for finance.
    """

    def __init__(self) -> None:
        pass

    def unit_converter(
        self,
        sparse: Optional[float] = None,
        dense: Optional[float] = None,
        factor: float = 1,
    ) -> float:
        """convert ``sparse`` or ``dense`` to each other using the rule of thump of ``dense = (factor) sparse``.

        Args:
            sparse (Optional[float], optional): the smaller/sparser amount which is a percentage of ``dense``,
                if provided calculates ``sparse = (factor) dense``.
            dense (Optional[float], optional): the larger/denser amount which is a multiplication of ``sparse``,
                if provided calculates ``dense = (factor) sparse``
            factor (float, optional): sparse to dense factor, either directly provided as a
                float number or as a predefined factor given by ``vizard.data.constant.FINANCIAL_RATIOS``

        Raises:
            ValueError: if ``sparse`` and ``dense`` are both None

        Returns:
            float: the converted amount
        """

        if sparse is not None:
            dense = factor * sparse
            return dense
        elif dense is not None:
            sparse = factor * dense
            return sparse
        else:
            raise ValueError("Only `sparse` or `dense` can be None.")


class FinancialUnitConverter(UnitConverter):
    """Contains utility tools for converting different financial units to each other.

    All functions that you implement should take the factor value using
    ``self.CONSTANTS['function_name']``. E.g.::

        def rent2deposit(self, rent: float) -> float:
            self.unit_converter(sparse=rent, dense=None,
                                factor=self.CONSTANTS['rent2deposit'])

    """

    def __init__(self, CONSTANTS: dict[str, float] = FINANCIAL_RATIOS) -> None:
        """Gets constant values needed for conversion

        Args:
            CONSTANTS (dict[str, float], optional): A dictionary of ``{string: float}`` where keys
                are function of this module and values are the factor used in conversion
                Defaults to ``vizard.data.constant.FINANCIAL_RATIOS``.
        """
        super().__init__()
        self.CONSTANTS = CONSTANTS

    def rent2deposit(self, rent: float) -> float:
        """Estimates deposit value via rent value by wrapping around :func:`unit_converter`

        Args:
            rent (float): rent amount (``rent`` is ``sparse``)

        Returns:
            float: deposit amount
        """
        return self.unit_converter(
            sparse=rent, dense=None, factor=self.CONSTANTS["rent2deposit"]
        )

    def deposit2rent(self, deposit: float) -> float:
        """Estimates rent value via deposit value by wrapping around :func:`unit_converter`

        Args:
            deposit (float): deposit amount (`deposit` is `dense`)

        Returns:
            float: rent amount
        """
        return self.unit_converter(
            sparse=None, dense=deposit, factor=self.CONSTANTS["deposit2rent"]
        )

    def deposit2worth(self, deposit: float) -> float:
        """Estimates worth value via deposit value by wrapping around :func:`unit_converter`

        Args:
            deposit (float): deposit amount (`deposit` is `sparse`)

        Returns:
            float: worth amount
        """
        return self.unit_converter(
            sparse=deposit, dense=None, factor=self.CONSTANTS["deposit2worth"]
        )

    def worth2deposit(self, worth: float) -> float:
        """Estimates deposit value via worth value by wrapping around :func:`unit_converter`

        Args:
            worth (float): worth amount (`worth` is `dense`)

        Returns:
            float: deposit amount
        """
        return self.unit_converter(
            sparse=None, dense=worth, factor=self.CONSTANTS["worth2deposit"]
        )

    def tax2income(self, tax: float) -> float:
        """Estimates income value via tax value by wrapping around :func:`unit_converter`

        Args:
            tax (float): tax amount (`tax` is `sparse`)

        Returns:
            float: income amount
        """
        return self.unit_converter(
            sparse=tax, dense=None, factor=self.CONSTANTS["tax2income"]
        )

    def income2tax(self, income: float) -> float:
        """Estimates tax value via income value by wrapping around :func:`unit_converter`

        Args:
            income (float): income amount (`income` is `dense`)

        Returns:
            float: tax amount
        """
        return self.unit_converter(
            sparse=None, dense=income, factor=self.CONSTANTS["income2tax"]
        )

    def income2worth(self, income: float) -> float:
        """Estimates worth value via income value by wrapping around :func:`unit_converter`

        Args:
            income (float): income amount (`income` is `sparse`)

        Returns:
            float: worth amount
        """
        return self.unit_converter(
            sparse=income, dense=None, factor=self.CONSTANTS["income2worth"]
        )

    def worth2income(self, worth: float) -> float:
        """Estimates income value via worth value by wrapping around :func:`unit_converter`

        Args:
            worth (float): worth amount (`worth` is `dense`)

        Returns:
            float: income amount
        """
        return self.unit_converter(
            sparse=None, dense=worth, factor=self.CONSTANTS["worth2income"]
        )


class WorldBankXMLProcessor:
    """An XML processor which is customized ot handle data dumped from https://data.worldbank.org/indicator

    Note:
        Since it is used by mainstream, works for us too.

    It's recommended to extend this class to work with particular indicator by
    first filtering by a "indicator", then manipulating the resulting dataframe.
    This can be done by calling :func:`indicator_filter`.

    Note:
        We prefer querying over ``Pandas`` dataframe than lxml_
        for processing and cleaning XML.

    .. _lxml: https://lxml.de/
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Args:
            dataframe: Main Pandas DataFrame to be processed
        """
        self.dataframe = dataframe

        # logging
        self.logger_name = ".WorldBankXMLProcessor"
        self.logger = logging.getLogger(logger.name + self.logger_name)

        # populate processed dict
        self.country_name_to_numeric_dict = self.indicator_filter()

    def indicator_filter(self) -> dict:
        """Aggregates using mean operation over all columns except name/index

        Values are scaled into [1, 7] range to match other
        world bank data processors.

        In this scenario, pivots a row-based dataframe to column based
        for ``'Years'`` and aggregates over them to achieve
        a 2-columns dataframe.

        Returns:
            dict:
                A dictionary of ``{string: float}`` where keys are country names
                and values are the corresponding indicator values.
        """
        dataframe = self.dataframe
        # pivot XML attributes of `<field>` tag
        dataframe = dataframe.pivot(columns="name", values="field")
        dataframe = dataframe.drop("Item", axis=1)
        # fill None s created by pivoting (one hot to stacked) only over country names
        dataframe["Country or Area"] = dataframe["Country or Area"].ffill().bfill()
        dataframe = dataframe.drop_duplicates()  # drop repetition of one hots
        dataframe = dataframe.ffill().bfill()  # fill None of values of countries
        dataframe = self.__include_years(dataframe=dataframe)  # exclude old years
        # dataframe = dataframe[dataframe['Year'].astype(int) >= 2017]
        dataframe = dataframe.drop_duplicates(
            subset=["Country or Area", "Year"], keep="last"
        ).reset_index()
        # pivot `Years` values as a set of separate columns i.e.
        #   from [name, years] -> [name, year1, year2, ...]
        df2 = dataframe.pivot(index="index", columns="Year", values="Value")
        # add names to pivoted years
        dataframe.drop("index", axis=1, inplace=True)
        dataframe.reset_index(inplace=True)
        df2.reset_index(inplace=True)
        dataframe = df2.join(dataframe["Country or Area"])
        # fill None s after pivoting `Years`
        country_names = dataframe["Country or Area"].unique()
        for cn in country_names:
            dataframe[dataframe["Country or Area"] == cn] = (
                dataframe[dataframe["Country or Area"] == cn].ffill().bfill()
            )
        # drop duplicates caused by filling None s of pivoting
        dataframe = dataframe.drop_duplicates(subset=["Country or Area"])

        # aggregation
        # drop scores/ranks and aggregate them into one column
        dataframe.drop("index", axis=1, inplace=True)
        mean_columns = [c for c in dataframe.columns.values if c.isnumeric()]
        dataframe["mean"] = dataframe[mean_columns].astype(float).mean(axis=1)
        dataframe.drop(dataframe.columns[:-2], axis=1, inplace=True)

        dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
            lambda x: x.lower()
        )

        # scale to [1-7] (standard of World Data Bank)
        column_max = dataframe["mean"].max()
        column_min = dataframe["mean"].min()

        def standardize(x):
            """Standardize the given value to [1, 7]"""
            return (((x - column_min) * (7.0 - 1.0)) / (column_max - column_min)) + 1.0

        dataframe["mean"] = dataframe["mean"].apply(standardize)
        return dict(
            zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]])
        )

    @staticmethod
    def __include_years(
        dataframe: pd.DataFrame,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> pd.DataFrame:
        """Drop columns to only include years given start and end.

        Note:
            Works inplace, hence manipulates original dataframe.

        # TODO:
            Currently only supports starting date

        Args:
            dataframe (:class:`pandas.DataFrame`): Pandas dataframe to be processed
            start (Optional[int], optional): start of years to include.
                Defaults to None.
            end (Optional[int], optional): end of years to include.
                Defaults to None.

        Returns:
            :class:`pandas.DataFrame`: A dataframe with a subset of years filtered on its columns.
        """
        start = 2017 if start is None else start

        assert end is None, "Currently only supports starting date"
        dataframe = dataframe[dataframe["Year"].astype(int) >= start]
        return dataframe

    def convert_country_name_to_numeric(self, string: Optional[str] = None) -> float:
        """Converts the name of a country into a numerical value

        If input ``string`` is None, uses the default value ``'Unknown'``. This
        is the hardcoded value in official form that we extract data from
        hence for consistency reasons, the same default value have been
        chosen.

        If the country name could not be found, then value of ``1.0`` will be
        returned. The reason for this is that we assume that the input country is
        not "famous" enough, hence we give lowest score in World Bank dataset,
        i.e. = 1.0.

        Args:
            string (Optional[str], optional): Name of a country

        Returns:
            float: Numerical value of the country
        """
        if string is None:
            string = "Unknown"
        string = string.lower()
        # see `self.indicator_filter` for description of `1.` and `150` magic numbers
        return functional.search_dict(
            string=string, if_nan=1.0, dic=self.country_name_to_numeric_dict
        )


class WorldBankDataframeProcessor:
    """A Pandas Dataframe processor which is customized to handle data dumped from WorldBank_

    It's recommended to extend this class to work with particular indicator by
    first filtering by a indicator_ from WorldBank_ , then manipulating
    the resulting dataframe.

    Note:
        Since it's used by mainstream, works for us too

    .. _WorldBank: https://govdata360.worldbank.org
    .. _indicator: https://data.worldbank.org/indicator
    """

    def __init__(
        self, dataframe: pd.DataFrame, subindicator_rank: Optional[bool] = False
    ) -> None:
        """Drops redundant columns of of `dataframe` and prepares a column wise subset of it

        args:
            dataframe (pd.DataFrame): Main Pandas DataFrame to be processed
            subindicator_rank (Optional[bool], optional): Whether or not use ranking (discrete)
                or score (continuous) for given ``indicator_name``. In original World Bank
                dataset, for some indicators, the score is discrete, while for others,
                it's continuous and this flag controls which one to extract.
                Defaults to False.
        """
        # set constants
        self.dataframe = dataframe
        self.INDICATOR = "Indicator"
        self.SUBINDICATOR = "Subindicator Type"
        self.subindicator_rank = subindicator_rank
        self.SUBINDICATOR_TYPE = "Rank" if subindicator_rank else "1-7 Best"
        # drop useless columns
        columns_to_drop = [
            "Country ISO3",
            "Indicator Id",
        ]
        columns_to_drop = columns_to_drop + [
            c for c in dataframe.columns.values if "-" in c
        ]
        self.dataframe.drop(columns_to_drop, axis=1, inplace=True)

        # logging
        self.logger_name = ".WorldBankDataframeProcessor"
        self.logger = logging.getLogger(logger.name + self.logger_name)

    def include_years(self, years: Tuple[Optional[int], Optional[int]] = None) -> None:
        """Processes a dataframe to only include years given tuple of ``years=(start, end)``.

        Works inplace, hence manipulates original dataframe.

        Args:
            years (Tuple[Optional[int], Optional[int]], optional): A tuple
                of ``(start, end)`` to limit years of data. If None,
                all years will be included. Defaults to None.
        """

        # figure out start and end year index of columns names values
        start_year, end_year = (
            [str(y) for y in years] if years is not None else (None, None)
        )
        column_years = [c for c in self.dataframe.columns.values if c.isnumeric()]
        start_year_index = (
            column_years.index(start_year) if start_year is not None else 0
        )
        end_year_index = column_years.index(end_year) if end_year is not None else -1
        # dataframe with desired years
        sub_column_years = column_years[start_year_index : end_year_index + 1]
        columns_to_drop = [
            c for c in list(set(column_years) - set(sub_column_years)) if c.isnumeric()
        ]
        self.dataframe.drop(columns_to_drop, axis=True, inplace=True)

    def indicator_filter(self, indicator_name: str) -> pd.DataFrame:
        """Filters the rows by given ``indicator_name`` and aggregates using mean operation

        Also, drops corresponding columns used for filtering.

        Args:
            indicator_name (string): A string containing an indicator's full name.
                See class level documents about available indicators.

        Returns:
            :class:`pandas.DataFrame`: A filtered dataframe with only a single ``indicator``.
        """
        # filter rows that only contain the provided `indicator_name` with type `rank` or `score`
        dataframe = self.dataframe.copy()
        dataframe = dataframe[
            (dataframe[self.INDICATOR] == indicator_name)
            & (dataframe[self.SUBINDICATOR] == self.SUBINDICATOR_TYPE)
        ]
        dataframe.drop([self.INDICATOR, self.SUBINDICATOR], axis=1, inplace=True)
        # drop scores/ranks and aggregate them into one column
        dataframe[indicator_name + "_mean"] = dataframe.mean(
            axis=1, skipna=True, numeric_only=True
        )
        dataframe.drop(dataframe.columns[1:-1], axis=1, inplace=True)

        # add a default row when input country name is 'Unknown` (this value was hardcoded in XFA PDF LOV field)
        df_unknown = pd.DataFrame(
            {dataframe.columns[0]: ["Unknown"], dataframe.columns[1]: [None]}
        )
        dataframe = pd.concat(
            objs=[dataframe, df_unknown],
            axis=0,
            verify_integrity=True,
            ignore_index=True,
        )

        # fillna since there is no info in the past years of that country -> unknown country
        if not self.subindicator_rank:  # fillna with lowest score = 1.
            dataframe = dataframe.fillna(value=1.0)
        else:  # fillna with highest rank = 150
            dataframe = dataframe.fillna(value=150)
        return dataframe


class EducationCountryScoreDataframePreprocessor(WorldBankDataframeProcessor):
    """Handles ``'Quality of the education system'`` indicator of a :class:`WorldBankDataframeProcessor` dataframe.

    The value ranges from 1 to 7 as score where higher is better.
    """

    def __init__(
        self, dataframe: pd.DataFrame, subindicator_rank: bool = False
    ) -> None:
        """See :class:`WorldBankDataframeProcessor` for more details."""
        super().__init__(dataframe, subindicator_rank)

        self.INDICATOR_NAME = "Quality of the education system, 1-7 (best)"
        self.country_name_to_numeric_dict = self.__indicator_filter()

        # logging
        self.logger_name = (
            ".WorldBankDataframeProcessor.EducationCountryScoreDataframePreprocessor"
        )
        self.logger = logging.getLogger(logger.name + self.logger_name)

    def __indicator_filter(self) -> dict:
        """Filters the rows by a constant ``INDICATOR_NAME``

        Returns:
            dict: A dictionary of ``country_name: score`` pairs.
        """
        dataframe = self.indicator_filter(indicator_name=self.INDICATOR_NAME)
        dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
            lambda x: x.lower()
        )
        return dict(
            zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]])
        )

    def convert_country_name_to_numeric(self, string: str) -> float:
        """Converts the name of a country into a numerical value

        If input `string` is None, uses the default value ``'Unknown'``. This
        is the hardcoded value in official form that we extract data from
        hence for consistency reasons, the same default value have been
        chosen.

        If the country name could not be found, then value of ``1.0`` will be
        returned as the **score**. The reason for this is that we assume that
        the input country is not "famous" enough, hence we give lowest **score**
        in World Bank dataset, i.e. = 1.0. Now, if instead of score, **rank** is
        chosen, then the worst rank, i.e. ``150`` will be returned.

        Args:
            string (str): Name of a country

        Returns:
            float: Numerical value of the country
        """

        if string is None:
            string = "Unknown"
        string = string.lower()
        # see `self.indicator_filter` for description of `1.` and `150` magic numbers
        return functional.search_dict(
            string=string,
            dic=self.country_name_to_numeric_dict,
            if_nan=1.0 if not self.subindicator_rank else 150,
        )


class EconomyCountryScoreDataframePreprocessor(WorldBankDataframeProcessor):
    """Handles ``'Global Competitiveness Index'`` indicator of a :class:`WorldBankDataframeProcessor` dataframe.

    The value ranges from 1 to 7 as the score where higher is better.
    """

    def __init__(
        self, dataframe: pd.DataFrame, subindicator_rank: bool = False
    ) -> None:
        """See :class:`WorldBankDataframeProcessor` for more details."""
        super().__init__(dataframe, subindicator_rank)

        self.INDICATOR_NAME = "Global Competitiveness Index"
        self.country_name_to_numeric_dict = self.__indicator_filter()

        # logging
        self.logger_name = (
            ".WorldBankDataframeProcessor.EconomyCountryScoreDataframePreprocessor"
        )
        self.logger = logging.getLogger(logger.name + self.logger_name)

    def __indicator_filter(self) -> dict:
        """Filters the rows by a constant ``INDICATOR_NAME``

        Returns:
            dict: A dictionary of ``country_name: score`` pairs.
        """
        dataframe = self.indicator_filter(indicator_name=self.INDICATOR_NAME)
        dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
            lambda x: x.lower()
        )
        return dict(
            zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]])
        )

    def convert_country_name_to_numeric(self, string: str) -> float:
        """Converts the name of a country into a numerical value

        If input `string` is None, uses the default value ``'Unknown'``. This
        is the hardcoded value in official form that we extract data from
        hence for consistency reasons, the same default value have been
        chosen.

        If the country name could not be found, then value of ``1.0`` will be
        returned as the **score**. The reason for this is that we assume that
        the input country is not "famous" enough, hence we give lowest **score**
        in World Bank dataset, i.e. = 1.0. Now, if instead of score, **rank** is
        chosen, then the worst rank, i.e. ``150`` will be returned.

        Args:
            string (str): Name of a country

        Returns:
            float: Numerical value of the country
        """
        if string is None:
            string = "Unknown"
        string = string.lower()
        # see `self.indicator_filter` for description of `1.` and `150` magic numbers
        return functional.search_dict(
            string=string,
            dic=self.country_name_to_numeric_dict,
            if_nan=1.0 if not self.subindicator_rank else 150,
        )


class CanadaDataframePreprocessor(DataframePreprocessor):
    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        super().__init__(dataframe)
        self.logger_name = ".CanadaDataframePreprocessor"
        self.logger = logging.getLogger(logger.name + self.logger_name)

        self.base_date = (
            None  # the time forms were filled, considered "today" for forms
        )

        # get country code to name dict
        self.config_path = CANADA_COUNTRY_CODE_TO_NAME
        self.CANADA_COUNTRY_CODE_TO_NAME = self.config_csv_to_dict(self.config_path)

    def convert_country_code_to_name(self, string: str) -> str:
        """
        Converts the (custom and non-standard) code of a country to its name given the XFA docs LOV section.
        # TODO: integrate this into `file_specific...` after verifying it in
        # `'notebooks/data_exploration_dev.ipynb'`

        Args:
            string: input code string
        """
        logger = logging.getLogger(self.logger.name + ".convert_country_code_to_name")

        country = [c for c in self.CANADA_COUNTRY_CODE_TO_NAME.keys() if string in c]
        if country:
            return self.CANADA_COUNTRY_CODE_TO_NAME[country]
        else:
            logger.debug(
                '"{}" country code could not be found in the config file="{}".'.format(
                    string, self.config_path
                )
            )
            return "Unknown"  # '000' code in XFA forms

    def file_specific_basic_transform(self, type: DOC_TYPES, path: str) -> pd.DataFrame:
        canada_xfa = CanadaXFA()  # Canada PDF to XML

        if type == DOC_TYPES.canada_5257e:
            # XFA to XML
            xml = canada_xfa.extract_raw_content(path)
            xml = canada_xfa.clean_xml_for_csv(xml=xml, type=DOC_TYPES.canada_5257e)
            # XML to flattened dict
            data_dict = canada_xfa.xml_to_flattened_dict(xml=xml)
            data_dict = canada_xfa.flatten_dict(data_dict)
            # clean flattened dict
            data_dict = functional.dict_summarizer(
                data_dict,
                cutoff_term=CANADA_CUTOFF_TERMS.ca5257e.value,
                KEY_ABBREVIATION_DICT=CANADA_5257E_KEY_ABBREVIATION,
                VALUE_ABBREVIATION_DICT=CANADA_5257E_VALUE_ABBREVIATION,
            )
            # convert each data dict to a dataframe
            dataframe = pd.DataFrame.from_dict(data=[data_dict], orient="columns")
            self.dataframe = dataframe
            # drop pepeg columns
            #   warning: setting `errors='ignore` ignores errors if columns do not exist!
            dataframe.drop(
                CANADA_5257E_DROP_COLUMNS, axis=1, inplace=True, errors="ignore"
            )

            # Adult binary state: adult=True or child=False
            dataframe["P1.AdultFlag"] = dataframe["P1.AdultFlag"].apply(
                lambda x: True if x == "adult" else False
            )
            # service language: 1=En, 2=Fr -> need to be changed to categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.ServiceIn.ServiceIn", dtype=np.int8, if_nan="skip"
            )
            # AliasNameIndicator: 1=True, 0=False
            dataframe[
                "P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator"
            ] = dataframe[
                "P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator"
            ].apply(
                lambda x: True if x == "Y" else False
            )
            # VisaType: String -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.VisaType.VisaType",
                dtype=str,
                if_nan="fill",
                value="OTHER",
            )
            # Birth City: String -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.PlaceBirthCity", dtype=str, if_nan="fill", value="OTHER"
            )
            # Birth country: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.PlaceBirthCountry",
                dtype=str,
                if_nan="fill",
                value="IRAN",
            )
            # citizen of: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.Citizenship.Citizenship",
                dtype=str,
                if_nan="fill",
                value="IRAN",
            )
            # current country of residency: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.CurrCOR.Row2.Country",
                dtype=str,
                if_nan="fill",
                value="IRAN",
            )
            # current country of residency status: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.CurrCOR.Row2.Status",
                dtype=np.int8,
                if_nan="fill",
                value=np.int8(6),  # 6=OTHER in the form
            )
            # current country of residency other description: bool -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.CurrCOR.Row2.Other",
                dtype=bool,
                if_nan="fill",
                value=False,
            )
            # validation date of information, i.e. current date: datetime
            dataframe = self.change_dtype(
                col_name="P3.Sign.C1CertificateIssueDate",
                dtype=parser.parse,
                if_nan="skip",
            )
            # keep it so we can access for other file if that was None
            if not dataframe["P3.Sign.C1CertificateIssueDate"].isna().all():
                self.base_date = dataframe["P3.Sign.C1CertificateIssueDate"]
            # date of birth in year: string -> datetime
            dataframe = self.change_dtype(
                col_name="P1.PD.DOBYear", dtype=parser.parse, if_nan="skip"
            )
            # current country of residency period: None -> Datetime (=age period)
            dataframe = self.change_dtype(
                col_name="P1.PD.CurrCOR.Row2.FromDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P1.PD.DOBYear"],
            )
            dataframe = self.change_dtype(
                col_name="P1.PD.CurrCOR.Row2.ToDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            # current country of residency period: Datetime -> int days
            dataframe = self.aggregate_datetime(
                col_base_name="P1.PD.CurrCOR.Row2",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=None,
            )
            # date of birth in year: datetime -> int days
            dataframe = self.aggregate_datetime(
                col_base_name="P1.PD.DOBYear",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=dataframe["P1.PD.DOBYear"],
                current_date=dataframe["P3.Sign.C1CertificateIssueDate"],
                one_sided="right",
            )
            # delete tnx to P1.PD.CurrCOR.Row2
            self.column_dropper(string="P1.PD.CORDates", inplace=True)
            # has previous country of residency: bool -> categorical
            dataframe["P1.PD.PCRIndicator"] = dataframe["P1.PD.PCRIndicator"].apply(
                lambda x: True if x == "Y" else False
            )

            # clean previous country of residency features
            country_tag_list = [
                c for c in dataframe.columns.values if "P1.PD.PrevCOR." in c
            ]
            PREV_COUNTRY_MAX_FEATURES = 4
            for i in range(len(country_tag_list) // PREV_COUNTRY_MAX_FEATURES):
                # in XLA extracted file, this section start from `Row2` (ie. i+2)
                i += 2
                # previous country of residency 02: string -> categorical
                dataframe = self.change_dtype(
                    col_name="P1.PD.PrevCOR.Row" + str(i) + ".Country",
                    dtype=str,
                    if_nan="fill",
                    value="OTHER",
                )
                # previous country of residency status 02: string -> categorical
                dataframe = self.change_dtype(
                    col_name="P1.PD.PrevCOR.Row" + str(i) + ".Status",
                    dtype=np.int8,
                    if_nan="fill",
                    value=np.int8(6),
                )
                # previous country of residency 02 period (P1.PD.PrevCOR.Row2): string -> datetime -> int days
                dataframe = self.change_dtype(
                    col_name="P1.PD.PrevCOR.Row" + str(i) + ".FromDate",
                    dtype=parser.parse,
                    if_nan="fill",
                    value=dataframe["P3.Sign.C1CertificateIssueDate"],
                )
                dataframe = self.change_dtype(
                    col_name="P1.PD.PrevCOR.Row" + str(i) + ".ToDate",
                    dtype=parser.parse,
                    if_nan="fill",
                    value=dataframe["P3.Sign.C1CertificateIssueDate"],
                )
                dataframe = self.aggregate_datetime(
                    col_base_name="P1.PD.PrevCOR.Row" + str(i),
                    type=DOC_TYPES.canada,
                    new_col_name="Period",
                    reference_date=None,
                    current_date=None,
                )
            # delete tnx to P1.PD.PrevCOR.Row2 and P1.PD.PrevCOR.Row3
            self.column_dropper(string="P1.PD.PCRDatesR", inplace=True)

            # apply from country of residency (cwa=country where apply): Y=True, N=False
            dataframe["P1.PD.SameAsCORIndicator"] = dataframe[
                "P1.PD.SameAsCORIndicator"
            ].apply(lambda x: True if x == "Y" else False)
            # country where applying: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.CWA.Row2.Country",
                dtype=str,
                if_nan="fill",
                value="OTHER",
            )
            # country where applying status: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.CWA.Row2.Status",
                dtype=np.int8,
                if_nan="fill",
                value=np.int8(6),
            )
            # country where applying other: string -> categorical
            dataframe = self.change_dtype(
                col_name="P1.PD.CWA.Row2.Other", dtype=bool, if_nan="fill", value=False
            )
            # country where applying period: datetime -> int days
            dataframe = self.change_dtype(
                col_name="P1.PD.CWA.Row2.FromDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.change_dtype(
                col_name="P1.PD.CWA.Row2.ToDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            # TODO: if None (here .Period=0): fill with average statistically
            dataframe = self.aggregate_datetime(
                col_base_name="P1.PD.CWA.Row2",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=None,
            )
            # delete tnx to P1.PD.CWA.Row2
            self.column_dropper(string="P1.PD.CWADates", inplace=True)
            # marriage period: datetime -> int days
            dataframe = self.change_dtype(
                col_name="P1.MS.SecA.DateOfMarr",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.aggregate_datetime(
                col_base_name="P1.MS.SecA.DateOfMarr",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=dataframe["P3.Sign.C1CertificateIssueDate"],
                one_sided="right",
            )
            # delete tnx to P1.MS.SecA.DateOfMarr
            self.column_dropper(string="P1.MS.SecA.MarrDate.From")
            # previous marriage: Y=True, N=False
            dataframe["P2.MS.SecA.PrevMarrIndicator"] = dataframe[
                "P2.MS.SecA.PrevMarrIndicator"
            ].apply(lambda x: True if x == "Y" else False)
            # previous marriage type of relationship
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.TypeOfRelationship",
                dtype=str,
                if_nan="fill",
                value="OTHER",
            )
            # previous spouse age period: string -> datetime -> int days
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.PrevSpouseDOB.DOBYear",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.aggregate_datetime(
                col_base_name="P2.MS.SecA.PrevSpouseDOB.DOBYear",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                one_sided="right",
                current_date=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            # previous marriage period: string -> datetime -> int days
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.FromDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.ToDate.ToDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.aggregate_datetime(
                col_base_name="P2.MS.SecA",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=None,
            )
            self.column_dropper(string="P2.MS.SecA.Prevly", inplace=True)
            # passport country of issue: string -> categorical
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.Psprt.CountryofIssue.CountryofIssue",
                dtype=str,
                if_nan="fill",
                value="OTHER",
            )
            # expiry remaining period: datetime -> int days
            # if None, fill with 1 year ago, ie. period=1year
            temp_date = dataframe["P3.Sign.C1CertificateIssueDate"].apply(
                lambda x: x + relativedelta(years=-1)
            )
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.Psprt.ExpiryDate",
                dtype=parser.parse,
                if_nan="fill",
                value=temp_date,
            )
            dataframe = self.aggregate_datetime(
                col_base_name="P2.MS.SecA.Psprt.ExpiryDate",
                type=DOC_TYPES.canada,
                new_col_name="Remaining",
                current_date=None,
                one_sided="left",
                reference_date=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            # native lang: string -> categorical
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.Langs.languages.nativeLang.nativeLang",
                dtype=str,
                if_nan="fill",
                value="IRAN",
            )
            # communication lang: Eng, Fr, both, none -> categorical
            dataframe = self.change_dtype(
                col_name="P2.MS.SecA.Langs.languages.ableToCommunicate.ableToCommunicate",
                dtype=str,
                if_nan="fill",
                value="NEITHER",
            )
            # language official test: bool -> binary
            dataframe["P2.MS.SecA.Langs.LangTest"] = dataframe[
                "P2.MS.SecA.Langs.LangTest"
            ].apply(lambda x: True if x == "Y" else False)
            # have national ID: bool -> binary
            dataframe["P2.natID.q1.natIDIndicator"] = dataframe[
                "P2.natID.q1.natIDIndicator"
            ].apply(lambda x: True if x == "Y" else False)
            # national ID country of issue: string -> categorical
            dataframe = self.change_dtype(
                col_name="P2.natID.natIDdocs.CountryofIssue.CountryofIssue",
                dtype=str,
                if_nan="fill",
                value="IRAN",
            )
            # United States doc: bool -> binary
            dataframe["P2.USCard.q1.usCardIndicator"] = dataframe[
                "P2.USCard.q1.usCardIndicator"
            ].apply(lambda x: True if x == "Y" else False)
            # drop contact information except having US Canada phone number
            self.column_dropper(string="P2.CI.cntct", exclude="CanadaUS", inplace=True)
            # US Canada phone number: bool -> binary
            dataframe["P2.CI.cntct.PhnNums.Phn.CanadaUS"] = dataframe[
                "P2.CI.cntct.PhnNums.Phn.CanadaUS"
            ].apply(lambda x: True if x == "1" else False)
            # US Canada alt phone number: bool -> binary
            dataframe["P2.CI.cntct.PhnNums.AltPhn.CanadaUS"] = dataframe[
                "P2.CI.cntct.PhnNums.AltPhn.CanadaUS"
            ].apply(lambda x: True if x == "1" else False)
            # purpose of visit: string, 8 states -> categorical
            dataframe = self.change_dtype(
                col_name="P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit",
                dtype=np.int8,
                if_nan="fill",
                value=np.int8(7),
            )  # 7 is other in the form
            # purpose of visit description: string -> binary
            dataframe = self.change_dtype(
                col_name="P3.DOV.PrpsRow1.Other.Other",
                dtype=bool,
                if_nan="fill",
                value=False,
            )
            # how long going to stay: None -> datetime (0 days)
            dataframe = self.change_dtype(
                col_name="P3.DOV.PrpsRow1.HLS.FromDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.change_dtype(
                col_name="P3.DOV.PrpsRow1.HLS.ToDate",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            # how long going to stay: datetime -> int days
            dataframe = self.aggregate_datetime(
                col_base_name="P3.DOV.PrpsRow1.HLS",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=None,
            )
            # delete tnx to P3.DOV.PrpsRow1.HLS
            self.column_dropper(
                string="P3.DOV.PrpsRow1.HLS", exclude="Period", inplace=True
            )
            # fund to integer
            dataframe = self.change_dtype(
                col_name="P3.DOV.PrpsRow1.Funds.Funds", dtype=np.int32, if_nan="skip"
            )
            # relation to applicant of purpose of visit 01: string -> categorical
            dataframe = self.change_dtype(
                col_name="P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe",
                dtype=str,
                if_nan="fill",
                value="OTHER",
            )
            # relation to applicant of purpose of visit 02: string -> categorical
            dataframe = self.change_dtype(
                col_name="P3.cntcts_Row2.Relationship.RelationshipToMe",
                dtype=str,
                if_nan="fill",
                value="OTHER",
            )
            # higher education: bool -> binary
            dataframe["P3.Edu.EduIndicator"] = dataframe["P3.Edu.EduIndicator"].apply(
                lambda x: True if x == "Y" else False
            )
            # higher education period: string -> datetime -> int days
            dataframe = self.change_dtype(
                col_name="P3.Edu.Edu_Row1.FromYear",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.change_dtype(
                col_name="P3.Edu.Edu_Row1.ToYear",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["P3.Sign.C1CertificateIssueDate"],
            )
            dataframe = self.aggregate_datetime(
                col_base_name="P3.Edu.Edu_Row1",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=None,
            )
            # higher education country: string -> categorical
            # TODO: skip here, and only fill with 'IRAN' for those who have 'P3.Edu.EduIndicator' = True
            #   see ### P3.Edu.Edu_Row1.Country.Country -> categorical in notebooks for more info
            dataframe = self.change_dtype(
                col_name="P3.Edu.Edu_Row1.Country.Country",
                dtype=str,
                if_nan="fill",
                value="IRAN",
            )
            # field of study: string -> categorical
            dataframe["P3.Edu.Edu_Row1.FieldOfStudy"] = dataframe[
                "P3.Edu.Edu_Row1.FieldOfStudy"
            ].astype("string")
            # clean occupation features
            occupation_tag_list = [
                c for c in dataframe.columns.values if "P3.Occ.OccRow" in c
            ]
            PREV_OCCUPATION_MAX_FEATURES = 4
            for i in range(len(occupation_tag_list) // PREV_OCCUPATION_MAX_FEATURES):
                i += 1  # in the form, it starts from Row1 (ie. i+1)
                # occupation period 01: none -> string year -> int days
                dataframe = self.change_dtype(
                    col_name="P3.Occ.OccRow" + str(i) + ".FromYear",
                    dtype=parser.parse,
                    if_nan="fill",
                    value=dataframe["P3.Sign.C1CertificateIssueDate"],
                )
                dataframe = self.change_dtype(
                    col_name="P3.Occ.OccRow" + str(i) + ".ToYear",
                    dtype=parser.parse,
                    if_nan="fill",
                    value=dataframe["P3.Sign.C1CertificateIssueDate"],
                )
                dataframe = self.aggregate_datetime(
                    col_base_name="P3.Occ.OccRow" + str(i),
                    type=DOC_TYPES.canada,
                    new_col_name="Period",
                    reference_date=None,
                    current_date=None,
                )
                # occupation type 01: string -> categorical
                dataframe = self.change_dtype(
                    col_name="P3.Occ.OccRow" + str(i) + ".Occ.Occ",
                    dtype=str,
                    if_nan="fill",
                    value="OTHER",
                )
                # occupation country: string -> categorical
                dataframe = self.change_dtype(
                    col_name="P3.Occ.OccRow" + str(i) + ".Country.Country",
                    dtype=str,
                    if_nan="fill",
                    value="IRAN",
                )

            # medical details: string -> binary
            dataframe = self.change_dtype(
                col_name="P3.BGI.Details.MedicalDetails",
                dtype=bool,
                if_nan="fill",
                value=False,
            )
            # other than medical: string -> binary
            dataframe = self.change_dtype(
                col_name="P3.BGI.otherThanMedic", dtype=bool, if_nan="fill", value=False
            )
            # without authentication stay, work, etc: bool -> binary
            dataframe["P3.noAuthStay"] = dataframe["P3.noAuthStay"].apply(
                lambda x: True if x == "Y" else False
            )
            # deported or refused entry: bool -> binary
            dataframe["P3.refuseDeport"] = dataframe["P3.refuseDeport"].apply(
                lambda x: True if x == "Y" else False
            )
            # previously applied: bool -> binary
            dataframe["P3.BGI2.PrevApply"] = dataframe["P3.BGI2.PrevApply"].apply(
                lambda x: True if x == "Y" else False
            )
            # criminal record: bool -> binary
            dataframe["P3.PWrapper.criminalRec"] = dataframe[
                "P3.PWrapper.criminalRec"
            ].apply(lambda x: True if x == "Y" else False)
            # military record: bool -> binary
            dataframe["P3.PWrapper.Military.Choice"] = dataframe[
                "P3.PWrapper.Military.Choice"
            ].apply(lambda x: True if x == "Y" else False)
            # political, violent movement record: bool -> binary
            dataframe["P3.PWrapper.politicViol"] = dataframe[
                "P3.PWrapper.politicViol"
            ].apply(lambda x: True if x == "Y" else False)
            # witness of ill treatment: bool -> binary
            dataframe["P3.PWrapper.witnessIllTreat"] = dataframe[
                "P3.PWrapper.witnessIllTreat"
            ].apply(lambda x: True if x == "Y" else False)
            # drop the time form was filled
            self.column_dropper(string="P3.Sign.C1CertificateIssueDate", inplace=True)

            return dataframe

        if type == DOC_TYPES.canada_5645e:
            # XFA to XML
            xml = canada_xfa.extract_raw_content(path)
            xml = canada_xfa.clean_xml_for_csv(xml=xml, type=DOC_TYPES.canada_5645e)
            # XML to flattened dict
            data_dict = canada_xfa.xml_to_flattened_dict(xml=xml)
            data_dict = canada_xfa.flatten_dict(data_dict)
            # clean flattened dict
            data_dict = functional.dict_summarizer(
                data_dict,
                cutoff_term=CANADA_CUTOFF_TERMS.ca5645e.value,
                KEY_ABBREVIATION_DICT=CANADA_5645E_KEY_ABBREVIATION,
                VALUE_ABBREVIATION_DICT=None,
            )

            # convert each data dict to a dataframe
            dataframe = pd.DataFrame.from_dict(data=[data_dict], orient="columns")
            self.dataframe = dataframe

            # drop pepeg columns
            #   warning: setting `errors='ignore` ignores errors if columns do not exist!
            dataframe.drop(
                CANADA_5645E_DROP_COLUMNS, axis=1, inplace=True, errors="ignore"
            )

            # transform multiple pleb columns into a single chad one and fixing column dtypes
            # type of application: (already one hot) string -> int
            cols = [col for col in dataframe.columns.values if "p1.Subform1" in col]
            for c in cols:
                dataframe = self.change_dtype(
                    col_name=c, dtype=np.int16, if_nan="fill", value=np.int16("0")
                )
            # drop all names
            self.column_dropper(string="Name", inplace=True)
            # drop all Accompany=No and only rely on Accompany=Yes using binary state
            self.column_dropper(string="No", inplace=True)
            # applicant marriage status: string to integer
            dataframe = self.change_dtype(
                col_name="p1.SecA.App.ChdMStatus",
                dtype=np.int16,
                if_nan="fill",
                value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value),
            )
            # validation date of information, i.e. current date: datetime
            dataframe = self.change_dtype(
                col_name="p1.SecC.SecCdate",
                dtype=parser.parse,
                if_nan="fill",
                value=self.base_date,
            )
            # spouse date of birth: string -> datetime
            dataframe = self.change_dtype(
                col_name="p1.SecA.Sps.SpsDOB",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["p1.SecC.SecCdate"],
            )
            # spouse age period: datetime -> int days
            dataframe = self.aggregate_datetime(
                col_base_name="p1.SecA.Sps.SpsDOB",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=dataframe["p1.SecC.SecCdate"],
                one_sided="right",
            )
            # spouse country of birth: string -> categorical
            dataframe = self.change_dtype(
                col_name="p1.SecA.Sps.SpsCOB", dtype=str, if_nan="skip"
            )
            # spouse occupation type (issue #2): string -> categorical
            dataframe = self.change_dtype(
                col_name="p1.SecA.Sps.SpsOcc", dtype=str, if_nan="fill", value="OTHER"
            )
            # spouse accompanying: coming=True or not_coming=False
            dataframe["p1.SecA.Sps.SpsAccomp"] = dataframe[
                "p1.SecA.Sps.SpsAccomp"
            ].apply(lambda x: True if x == "1" else False)
            # mother date of birth: string -> datetime
            dataframe = self.change_dtype(
                col_name="p1.SecA.Mo.MoDOB",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["p1.SecC.SecCdate"],
            )
            # mother age period: datetime -> int days
            # TODO: if =0 (originally None), fill with average age difference of parents and children
            dataframe = self.aggregate_datetime(
                col_base_name="p1.SecA.Mo.MoDOB",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=dataframe["p1.SecC.SecCdate"],
                one_sided="right",
            )
            # mother occupation type (issue #2): string -> categorical
            dataframe = self.change_dtype(
                col_name="p1.SecA.Mo.MoOcc", dtype=str, if_nan="fill", value="OTHER"
            )
            # mother marriage status: int -> categorical
            dataframe = self.change_dtype(
                col_name="p1.SecA.Mo.ChdMStatus",
                dtype=np.int16,
                if_nan="fill",
                value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value),
            )
            # mother accompanying: coming=True or not_coming=False
            dataframe["p1.SecA.Mo.MoAccomp"] = dataframe["p1.SecA.Mo.MoAccomp"].apply(
                lambda x: True if x == "1" else False
            )
            # father date of birth: string -> datetime
            dataframe = self.change_dtype(
                col_name="p1.SecA.Fa.FaDOB",
                dtype=parser.parse,
                if_nan="fill",
                value=dataframe["p1.SecC.SecCdate"],
            )
            # father age period: datetime -> int days
            # TODO: if =0 (originally None), fill with average age difference of parents and children
            dataframe = self.aggregate_datetime(
                col_base_name="p1.SecA.Fa.FaDOB",
                type=DOC_TYPES.canada,
                new_col_name="Period",
                reference_date=None,
                current_date=dataframe["p1.SecC.SecCdate"],
                one_sided="right",
            )
            # mother occupation type (issue #2): string -> categorical
            dataframe = self.change_dtype(
                col_name="p1.SecA.Fa.FaOcc", dtype=str, if_nan="fill", value="OTHER"
            )
            # father marriage status: int -> categorical
            dataframe = self.change_dtype(
                col_name="p1.SecA.Fa.ChdMStatus",
                dtype=np.int16,
                if_nan="fill",
                value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value),
            )
            # father accompanying: coming=True or not_coming=False
            dataframe["p1.SecA.Fa.FaAccomp"] = dataframe["p1.SecA.Fa.FaAccomp"].apply(
                lambda x: True if x == "1" else False
            )

            # children's status
            children_tag_list = [
                c for c in dataframe.columns.values if "p1.SecB.Chd" in c
            ]
            CHILDREN_MAX_FEATURES = 7
            for i in range(len(children_tag_list) // CHILDREN_MAX_FEATURES):
                # child's marriage status 01: string to integer
                dataframe = self.change_dtype(
                    col_name="p1.SecB.Chd.[" + str(i) + "].ChdMStatus",
                    dtype=np.int16,
                    if_nan="fill",
                    value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value),
                )
                # child's relationship 01: string -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecB.Chd.[" + str(i) + "].ChdRel",
                    dtype=str,
                    if_nan="fill",
                    value="OTHER",
                )
                # child's date of birth 01: string -> datetime
                dataframe = self.change_dtype(
                    col_name="p1.SecB.Chd.[" + str(i) + "].ChdDOB",
                    dtype=parser.parse,
                    if_nan="skip",
                )
                # child's age period 01: datetime -> int days
                dataframe = self.aggregate_datetime(
                    type=DOC_TYPES.canada,
                    col_base_name="p1.SecB.Chd.[" + str(i) + "].ChdDOB",
                    new_col_name="Period",
                    reference_date=None,
                    one_sided="right",
                    current_date=dataframe["p1.SecC.SecCdate"],
                    if_nan="skip",
                )
                # child's country of birth 01: string -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecB.Chd.[" + str(i) + "].ChdCOB",
                    dtype=str,
                    if_nan="fill",
                    value="IRAN",
                )
                # child's occupation type 01 (issue #2): string -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecB.Chd.[" + str(i) + "].ChdOcc",
                    dtype=str,
                    if_nan="fill",
                    value="OTHER",
                )
                # child's marriage status: int -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecB.Chd.[" + str(i) + "].ChdMStatus",
                    dtype=np.int16,
                    if_nan="fill",
                    value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value),
                )
                # child's accompanying 01: coming=True or not_coming=False
                dataframe["p1.SecB.Chd.[" + str(i) + "].ChdAccomp"] = dataframe[
                    "p1.SecB.Chd.[" + str(i) + "].ChdAccomp"
                ].apply(lambda x: True if x == "1" else False)

                # check if the child does not exist and fill it properly (ghost case monkaS)
                if (
                    (
                        dataframe["p1.SecB.Chd.[" + str(i) + "].ChdMStatus"]
                        == CANADA_FILLNA.ChdMStatus_5645e.value
                    ).all()
                    and (
                        dataframe["p1.SecB.Chd.[" + str(i) + "].ChdRel"] == "OTHER"
                    ).all()
                    and (dataframe["p1.SecB.Chd.[" + str(i) + "].ChdDOB"].isna()).all()
                    and (
                        dataframe["p1.SecB.Chd.[" + str(i) + "].ChdAccomp"] == False
                    ).all()
                ):
                    # ghost child's date of birth: None -> datetime (current date) -> 0 days
                    dataframe = self.change_dtype(
                        col_name="p1.SecB.Chd.[" + str(i) + "].ChdDOB",
                        dtype=parser.parse,
                        if_nan="fill",
                        value=dataframe["p1.SecC.SecCdate"],
                    )
                    # ghost child's age period: datetime (current date) -> int 0 days
                    dataframe = self.aggregate_datetime(
                        type=DOC_TYPES.canada,
                        col_base_name="p1.SecB.Chd.[" + str(i) + "].ChdDOB",
                        new_col_name="Period",
                        reference_date=None,
                        one_sided="right",
                        current_date=dataframe["p1.SecC.SecCdate"],
                        if_nan=None,
                    )

            # fill existing child's date of birth where it is None with a heuristic
            # take average age period of children
            col_names = []  # holds all age periods
            col_names_age_all = []  # holds all age periods and date of births
            for i in range(len(children_tag_list) // CHILDREN_MAX_FEATURES):
                col_name = "p1.SecB.Chd.[" + str(i) + "].ChdDOB.Period"
                if col_name in dataframe.columns.values:
                    col_names.append(col_name)
                col_name = "p1.SecB.Chd.[" + str(i) + "].ChdDOB"
                if col_name in dataframe.columns.values:
                    col_names_age_all.append(col_name)
            # extract `Chd.DOB` from `Chd.DOB.Period`
            col_names_unprocessed = list(set(col_names_age_all) - set(col_names))
            for c in col_names_unprocessed:  # drop columns after processing them
                # average of family children as the heuristic
                dataframe[c + ".Period"] = dataframe[dataframe[col_names] != 0].mean(
                    axis=1, numeric_only=True
                )
                dataframe.drop(c, axis=1, inplace=True)

            # siblings' status
            siblings_tag_list = [
                c for c in dataframe.columns.values if "p1.SecC.Chd" in c
            ]
            SIBLINGS_MAX_FEATURES = 7
            for i in range(len(siblings_tag_list) // SIBLINGS_MAX_FEATURES):
                # sibling's marriage status 01: string to integer
                dataframe = self.change_dtype(
                    col_name="p1.SecC.Chd.[" + str(i) + "].ChdMStatus",
                    dtype=np.int16,
                    if_nan="fill",
                    value=np.int16(CANADA_FILLNA.ChdMStatus_5645e.value),
                )
                # sibling's relationship 01: string -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecC.Chd.[" + str(i) + "].ChdRel",
                    dtype=str,
                    if_nan="fill",
                    value="OTHER",
                )
                # sibling's date of birth 01: string -> datetime
                dataframe = self.change_dtype(
                    col_name="p1.SecC.Chd.[" + str(i) + "].ChdDOB",
                    dtype=parser.parse,
                    if_nan="skip",
                )
                # sibling's age period 01: datetime -> int days
                dataframe = self.aggregate_datetime(
                    type=DOC_TYPES.canada,
                    col_base_name="p1.SecC.Chd.[" + str(i) + "].ChdDOB",
                    new_col_name="Period",
                    reference_date=None,
                    one_sided="right",
                    current_date=dataframe["p1.SecC.SecCdate"],
                    if_nan="skip",
                )
                # sibling's country of birth 01: string -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecC.Chd.[" + str(i) + "].ChdCOB",
                    dtype=str,
                    if_nan="fill",
                    value="IRAN",
                )
                # sibling's occupation type 01 (issue #2): string -> categorical
                dataframe = self.change_dtype(
                    col_name="p1.SecC.Chd.[" + str(i) + "].ChdOcc",
                    dtype=str,
                    if_nan="fill",
                    value="OTHER",
                )
                # sibling's accompanying: coming=True or not_coming=False
                dataframe["p1.SecC.Chd.[" + str(i) + "].ChdAccomp"] = dataframe[
                    "p1.SecC.Chd.[" + str(i) + "].ChdAccomp"
                ].apply(lambda x: True if x == "1" else False)

                # check if the sibling does not exist and fill it properly (ghost case monkaS)
                if (
                    (
                        dataframe["p1.SecC.Chd.[" + str(i) + "].ChdMStatus"]
                        == CANADA_FILLNA.ChdMStatus_5645e.value
                    ).all()
                    and (
                        dataframe["p1.SecC.Chd.[" + str(i) + "].ChdRel"] == "OTHER"
                    ).all()
                    and (dataframe["p1.SecC.Chd.[" + str(i) + "].ChdOcc"].isna()).all()
                    and (
                        dataframe["p1.SecC.Chd.[" + str(i) + "].ChdAccomp"] == False
                    ).all()
                ):
                    # ghost sibling's date of birth: None -> datetime (current date) -> 0 days
                    dataframe = self.change_dtype(
                        col_name="p1.SecC.Chd.[" + str(i) + "].ChdDOB",
                        dtype=parser.parse,
                        if_nan="fill",
                        value=dataframe["p1.SecC.SecCdate"],
                    )
                    # ghost sibling's age period: datetime (current date) -> int 0 days
                    dataframe = self.aggregate_datetime(
                        type=DOC_TYPES.canada,
                        col_base_name="p1.SecC.Chd.[" + str(i) + "].ChdDOB",
                        new_col_name="Period",
                        reference_date=None,
                        one_sided="right",
                        current_date=dataframe["p1.SecC.SecCdate"],
                        if_nan=None,
                    )

            # fill existing sibling's date of birth where it is None with a heuristic
            # take average age period of siblings
            col_names = []  # holds all age periods
            col_names_age_all = []  # holds all age periods and date of births
            for i in range(len(siblings_tag_list) // SIBLINGS_MAX_FEATURES):
                col_name = "p1.SecC.Chd.[" + str(i) + "].ChdDOB.Period"
                if col_name in dataframe.columns.values:
                    col_names.append(col_name)
                col_name = "p1.SecC.Chd.[" + str(i) + "].ChdDOB"
                if col_name in dataframe.columns.values:
                    col_names_age_all.append(col_name)
            # extract `Chd.DOB` from `Chd.DOB.Period`
            col_names_unprocessed = list(set(col_names_age_all) - set(col_names))
            for c in col_names_unprocessed:  # drop columns after processing them
                # average of family siblings as the heuristic
                dataframe[c + ".Period"] = dataframe[dataframe[col_names] != 0].mean(
                    axis=1, numeric_only=True
                )
                dataframe.drop(c, axis=1, inplace=True)

            # drop the time form was filled
            self.column_dropper(string="p1.SecC.SecCdate", inplace=True)

            return dataframe

        if type == DOC_TYPES.canada_label:
            dataframe = pd.read_csv(path, sep=" ", names=["VisaResult"])
            functional.change_dtype(
                dataframe=dataframe,
                col_name="VisaResult",
                dtype=np.int8,
                if_nan="fill",
                value=np.int8(0),
            )
            return dataframe


class FileTransform:
    """A base class for applying transforms as a composable object over files.

    Any behavior over the files itself (not the content of files)
    must extend this class.

    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(logger.name + ".FileTransform")
        pass

    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        """

        Args:
            src: source file to be processed
            dst: the pass that the processed file to be saved
        """
        pass


class CopyFile(FileTransform):
    """Only copies a file, a wrapper around `shutil`'s copying methods

    Default is set to 'cf', i.e. `shutil.copyfile`. For more info see
    shutil_ documentation.


    Reference:
        1. https://stackoverflow.com/a/30359308/18971263
    """

    def __init__(self, mode: str) -> None:
        super().__init__()

        self.COPY_MODES = ["c", "cf", "c2"]
        self.mode = mode if mode is not None else "cf"
        self.__check_mode(mode=mode)

    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        if self.mode == "c":
            shutil.copy(src=src, dst=dst)
        elif self.mode == "cf":
            shutil.copyfile(src=src, dst=dst)
        elif self.mode == "c2":
            shutil.copy2(src=src, dst=dst)

    def __check_mode(self, mode: str):
        """Checks copying mode to be available in shutil_

        Args:
            mode: copying mode in `shutil`, one of `'c'`, `'cf'`, `'c2'`

        .. _shutil: https://docs.python.org/3/library/shutil.html
        """
        if not mode in self.COPY_MODES:
            raise ValueError(
                (f"Mode {mode} does not exist,", f'choose one of "{self.COPY_MODES}".')
            )


class MakeContentCopyProtectedMachineReadable(FileTransform):
    """Reads a 'content-copy' protected PDF and removes this restriction

    Removing the protection is done by saving a "printed" version of via pikepdf_

    References:
        1. https://www.reddit.com/r/Python/comments/t32z2o/simple_code_to_unlock_all_readonly_pdfs_in/
        2. https://pikepdf.readthedocs.io/en/latest/

    .. _pikepdf: https://pikepdf.readthedocs.io/en/latest/
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        """

        Args:
            src (str): source file to be processed
            dst (str): destination to save the processed file

        Returns:
            Any: None
        """
        pdf = pikepdf.open(src, allow_overwriting_input=True)
        pdf.save(dst)


class FileTransformCompose:
    """Composes several transforms operating on files together

    The transforms should be tied to files with keyword and this will be only applying
    functions on files that match the keyword using a dictionary

    Transformation dictionary over files in the following structure::

        {
            FileTransform: 'filter_str',
            ...,
        }

    Note:
        Transforms will be applied in order of the keys in the dictionary
    """

    def __init__(self, transforms: dict) -> None:
        """

        Args:
            transforms: a dictionary of transforms, where the key is the instance of
                FileTransform and the value is the keyword that the transform will be
                applied to

        Raises:
            ValueError: if the keyword is not a string
        """
        if transforms is not None:
            for k in transforms.keys():
                if not issubclass(k.__class__, FileTransform):
                    raise TypeError(f"Keys must be {FileTransform} instance.")

        self.transforms = transforms

    def __call__(self, src: str, dst: str, *args: Any, **kwds: Any) -> Any:
        """Applies transforms in order

        Args:
            src (str): source file path to be processed
            dst (str): destination to save the processed file
        """
        for transform, file_filter in self.transforms.items():
            if file_filter in src:
                transform(src, dst)
