"""
Contains implementation of functions that could be used for processing data everywhere and
    are not necessarily bounded to a class.

"""
import re
import csv
import pandas as pd
import numpy as np
from dateutil import parser
from typing import List, Union
from types import FunctionType

from utils.constant import DOC_TYPES


def dict_summarizer(d: dict, cutoff_term: str, KEY_ABBREVIATION_DICT: dict = None,
                    VALUE_ABBREVIATION_DICT: dict = None) -> dict:
    """
    Takes a flattened dictionary and modifies it's keys in a way to shorten them by throwing away
        some part and using a abbreviation dictionary for both keys and values.

    args:
        d: The dictionary to be shortened
        cutoff_term: The string that used to find in keys and remove anything behind it
        KEY_ABBREVIATION_DICT: A dictionary containing abbreviation mapping for keys
        VALUE_ABBREVIATION_DICT: A dictionary containing abbreviation mapping for values 
    """

    new_keys = {}
    new_values = {}
    for k, v in d.items():
        if KEY_ABBREVIATION_DICT is not None:
            new_k = k
            if cutoff_term in k:  # FIXME: cutoff part should be outside of abbreviation
                new_k = k[k.index(cutoff_term)+len(cutoff_term)+1:]

            # add any filtering over keys here
            # abbreviation
            for word, abbr in KEY_ABBREVIATION_DICT.items():
                new_k = re.sub(word, abbr, new_k)
            new_keys[k] = new_k

        if VALUE_ABBREVIATION_DICT is not None:
            # values can be `None`
            if v is not None:
                new_v = v
                if cutoff_term in v:  # FIXME: cutoff part should be outside of abbreviation
                    new_v = v[v.index(cutoff_term)+len(cutoff_term)+1:]

                # add any filtering over values here
                # abbreviation
                for word, abbr in VALUE_ABBREVIATION_DICT.items():
                    new_v = re.sub(word, abbr, new_v)
                new_values[v] = new_v
            else:
                new_values[v] = v

    # return a new dictionary with updated values
    if KEY_ABBREVIATION_DICT is None:
        new_keys = dict((key, key) for (key, _) in d.items())
    if VALUE_ABBREVIATION_DICT is None:
        new_values = dict((value, value) for (_, value) in d.items())
    return dict((new_keys[key], new_values[value]) for (key, value) in d.items())


def dict_to_csv(d: dict, path: str) -> None:
    """
    Takes a flattened dictionary and writes it to a CSV file.

    args:
        d: A dictionary
        path: String to where file should be saved
    """

    with open(path, 'w') as f:
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)


def column_dropper(dataframe: pd.DataFrame, string: str, exclude: str = None,
                   regex: bool = False, inplace: bool = True) -> Union[None, pd.DataFrame]:
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

    if regex:
        r = re.compile(string)
        col_to_drop = list(filter(r.match, dataframe.columns.values))
    else:
        col_to_drop = [
            col for col in dataframe.columns.values if string in col]

    if exclude is not None:
        col_to_drop = [col for col in col_to_drop if exclude not in col]

    if inplace:
        dataframe.drop(col_to_drop, axis=1, inplace=inplace)
    else:
        dataframe = dataframe.drop(col_to_drop, axis=1, inplace=inplace)

    return None if inplace else dataframe


def fillna_datetime(dataframe: pd.DataFrame, col_base_name: str, date: str, type: DOC_TYPES,
                    one_sided: str = False, inplace: bool = False) -> Union[None, pd.DataFrame]:
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
        inplace: whether or not use an inplace operation 
        type: `DOC_TYPE` used to use rules for matching tags and filling appropriately
    """

    if not one_sided:
        r = re.compile(tag_to_regex_compatible(
            string=col_base_name, type=type))
    else:
        r = re.compile(tag_to_regex_compatible(
            string=col_base_name, type=type)+'\.(From|To).+')
    columns_to_fillna_names = list(filter(r.match, dataframe.columns.values))
    for col in dataframe[columns_to_fillna_names]:
        if inplace:
            dataframe[col].fillna(date, inplace=inplace)
        else:
            dataframe[col] = dataframe[col].fillna(date, inplace=inplace)
    return None if inplace else dataframe


def aggregate_datetime(dataframe: pd.DataFrame, col_base_name: str, new_col_name: str,
                       type: DOC_TYPES, if_nan: Union[str, FunctionType] = 'skip',
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
        type: `DOC_TYPE` used to use rules for matching tags and filling appropriately
        if_nan: What to do with `None`s (NaN). Could be a function or predfined states as follow:\n
            1. 'skip': do nothing (i.e. ignore `None`'s)
    """

    aggregated_column_name = None
    if one_sided is None:
        aggregated_column_name = col_base_name + '.' + new_col_name
        r = re.compile(tag_to_regex_compatible(
            string=col_base_name, type=type)+'\.(From|To).+')
    else:  # when one_sided, we no longer have *From* or *To*
        aggregated_column_name = col_base_name + '.' + new_col_name
        r = re.compile(tag_to_regex_compatible(
            string=col_base_name, type=type))
    columns_to_aggregate_names = list(
        filter(r.match, dataframe.columns.values))

    # *.FromDate and *.ToDate --> *.Period
    column_from_date = reference_date
    column_to_date = current_date
    if one_sided == 'left':
        column_from_date = reference_date
        to_date = columns_to_aggregate_names[0]
    elif one_sided == 'right':
        column_to_date = current_date
        from_date = columns_to_aggregate_names[0]
    else:
        from_date = [
            col for col in columns_to_aggregate_names if 'From' in col][0]
        to_date = [col for col in columns_to_aggregate_names if 'To' in col][0]

    if isinstance(column_to_date, str):
        column_to_date = parser.parse(column_to_date)

    if column_from_date is None:  # ignore reference_date if from_date exists
        # to able to use already parsed data from fillna
        if not dataframe[from_date].dtypes == '<M8[ns]':
            dataframe[from_date] = dataframe[from_date].apply(
                lambda x: parser.parse(x) if x is not None else x)
        column_from_date = dataframe[from_date]
    else:
        if isinstance(column_from_date, str):
            column_from_date = parser.parse(column_from_date)

    if column_to_date is None:  # ignore current_date if to_date exists
        # to able to use already parsed data from fillna
        if not dataframe[to_date].dtypes == '<M8[ns]':
            dataframe[to_date] = dataframe[to_date].apply(
                lambda x: parser.parse(x) if x is not None else x)
        column_to_date = dataframe[to_date]
    else:
        if isinstance(column_to_date, str):
            column_to_date = parser.parse(column_to_date)

    if if_nan == 'skip':
        if column_from_date.isna().all() or column_to_date.isna().all():
            return dataframe

    dataframe[aggregated_column_name] = np.nan  # combination of dates
    dataframe[aggregated_column_name].fillna(
        column_to_date - column_from_date, inplace=True)  # period
    dataframe[aggregated_column_name] = dataframe[aggregated_column_name].dt.days.astype(
        'int32')  # change to int of days

    dataframe.drop(columns_to_aggregate_names, axis=1,
                   inplace=True)  # drop from/to columns
    return dataframe


def tag_to_regex_compatible(string: str, type: DOC_TYPES) -> str:
    """
    Takes a string and makes it regex compatible for 

    args:
        string: input string to get manipulated
        type: specified `DOC_TYPE` to determine regex rules 
    """

    if type == DOC_TYPES.canada_5257e or type == DOC_TYPES.canada_5645e or type == DOC_TYPES.canada:
        string = string.replace('.', '\.').replace(
            '[', '\[').replace(']', '\]')

    return string


def change_dtype(dataframe: pd.DataFrame, col_name: str, dtype: FunctionType,
                 if_nan: Union[str, FunctionType] = 'skip', **kwargs):
    """
    Takes a column name and changes the dataframe's column data type where for 
        None (nan) values behave based on `if_nan` argument.

    args:
        col_name: Column name of the dataframe
        dtype: target data type as a function e.g. `np.float32`
        if_nan: What to do with `None`s (NaN). Could be a function or predfined states as follow:\n
            1. 'skip': do nothing (i.e. ignore `None`'s)
            2. 'value': fill the `None` with `value` argument via `kwargs`
    """

    # the function to be used in `.apply` method of dataframe
    func = None if isinstance(if_nan, str) else if_nan

    # define `func` for different cases of predfined logics
    if isinstance(if_nan, str):  # predefined `if_nan` cases
        if if_nan == 'skip':
            def func(x): return x
        if if_nan == 'fill':
            value = kwargs['value']
            assert isinstance(value, dtype)  # 'fill' `value` must be of type `type` 
            def func(x): return value 

    # apply the rules and data type change

    dataframe[col_name] = dataframe[col_name].apply(
        lambda x: dtype(x) if x is not None else func(x))

    return dataframe
