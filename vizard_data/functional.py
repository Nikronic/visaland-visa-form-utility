__all__ = ['dict_summarizer', 'dict_to_csv', 'column_dropper', 'fillna_datetime', 'aggregate_datetime',
           'tag_to_regex_compatible', 'change_dtype', 'unit_converter', 'flatten_dict',
           'xml_to_flattened_dict', 'create_directory_structure_tree', 'dump_directory_structure_csv',
           'process_directory', 'search_dict', 'config_csv_to_dict']

"""
Contains implementation of functions that could be used for processing data everywhere and
    are not necessarily bounded to a class.

"""

import re
import os
import csv
import xmltodict
import pandas as pd
import numpy as np
from dateutil import parser
import datetime
from fnmatch import fnmatch
from typing import Any, Callable, List, Union
import logging

from vizard_data.constant import DOC_TYPES
from vizard_data.constant import DATEUTIL_DEFAULT_DATETIME

from vizard_utils.helpers import loggingdecorator
from vizard_data.preprocessor import FileTransformCompose


# set logger
logger = logging.getLogger('__main__')


@loggingdecorator(logger.name+'.func', level=logging.DEBUG, output=False)
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


@loggingdecorator(logger.name+'.func', level=logging.INFO, output=False)
def dict_to_csv(d: dict, path: str) -> None:
    """
    Takes a flattened dictionary and writes it to a CSV file.

    args:
        d: A dictionary
        path: Path to the output file (will be created if not exist)
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
                    one_sided: Union[str, bool] = False, inplace: bool = False) -> Union[None, pd.DataFrame]:
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
                       type: DOC_TYPES, if_nan: Union[str, Callable, None] = 'skip',
                       one_sided: str = None, reference_date: str = None,
                       current_date: str = None, **kwargs) -> pd.DataFrame:
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
        if_nan: What to do with `None`s (NaN). Could be a function or predefined states as follow:\n
            1. 'skip': do nothing (i.e. ignore `None`'s)
        default_datetime: accepts `datetime.datetime` to set default date for `dtype=parser.parse`
    """

    default_datetime = datetime.datetime(year=DATEUTIL_DEFAULT_DATETIME['year'],
                                         month=DATEUTIL_DEFAULT_DATETIME['month'],
                                         day=DATEUTIL_DEFAULT_DATETIME['day'])
    default_datetime = kwargs.get('default_datetime', default_datetime)

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
        column_to_date = parser.parse(column_to_date, default=default_datetime)  # type: ignore

    if column_from_date is None:  # ignore reference_date if from_date exists
        # to able to use already parsed data from fillna
        if not dataframe[from_date].dtypes == '<M8[ns]':
            dataframe[from_date] = dataframe[from_date].apply(
                lambda x: parser.parse(x, default=default_datetime) if x is not None else x)
        column_from_date = dataframe[from_date]
    else:
        if isinstance(column_from_date, str):
            column_from_date = parser.parse(column_from_date, default=default_datetime)  # type: ignore

    if column_to_date is None:  # ignore current_date if to_date exists
        # to able to use already parsed data from fillna
        if not dataframe[to_date].dtypes == '<M8[ns]':
            dataframe[to_date] = dataframe[to_date].apply(
                lambda x: parser.parse(x, default=default_datetime) if x is not None else x)
        column_to_date = dataframe[to_date]
    else:
        if isinstance(column_to_date, str):
            column_to_date = parser.parse(column_to_date, default=default_datetime)  # type: ignore

    if if_nan is not None:
        if if_nan == 'skip':
            if column_from_date.isna().all() or column_to_date.isna().all():  # type: ignore
                return dataframe

    dataframe[aggregated_column_name] = np.nan  # combination of dates
    dataframe[aggregated_column_name].fillna(  # period
        column_to_date - column_from_date, inplace=True)  # type: ignore
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


def change_dtype(dataframe: pd.DataFrame, col_name: str, dtype: Callable,
                 if_nan: Union[str, Callable] = 'skip', **kwargs):
    """
    Takes a column name and changes the dataframe's column data type where for 
        None (nan) values behave based on `if_nan` argument.

    args:
        col_name: Column name of the dataframe
        dtype: target data type as a function e.g. `np.float32`
        if_nan: What to do with `None`s (NaN). Could be a function or predefined states as follow:\n
            1. 'skip': do nothing (i.e. ignore `None`'s)
            2. 'value': fill the `None` with `value` argument via `kwargs`
        default_datetime: accepts `datetime.datetime` to set default date for `dtype=parser.parse`
    """

    default_datetime = datetime.datetime(year=DATEUTIL_DEFAULT_DATETIME['year'],
                                         month=DATEUTIL_DEFAULT_DATETIME['month'],
                                         day=DATEUTIL_DEFAULT_DATETIME['day'])
    default_datetime = kwargs.get('default_datetime', default_datetime)

    # define `func` for different cases of predefined logics
    if isinstance(if_nan, str):  # predefined `if_nan` cases
        if if_nan == 'skip':
            # the function to be used in `.apply` method of dataframe
            def func(x): return x
        elif if_nan == 'fill':
            value = kwargs['value']
            # the function to be used in `.apply` method of dataframe
            def func(x): return value
        else:
            raise ValueError('Unknown mode "{}".'.format(if_nan))
    else:
        pass

    def standardize(value: Any):
        """
        Takes a value and make it standard for the target function that is going to parse it

        Remark: This is mostly hardcoded and cannot be written better (I think!). So, you can
            remove it entirely, and see what errors you get, and change this accordingly to 
            errors and exceptions you get.

        args:
            value: the input value that need to be standardized
        """
        if dtype == parser.parse:  # datetime parser
            try:
                parser.parse(value)
            except ValueError:  # bad input format for `parser.parse`
                # we want YYYY-MM-DD
                # MMDDYYYY format (Canada Common Forms)
                if len(value) == 8 and value.isnumeric():
                    value = '{}-{}-{}'.format(value[4:],
                                              value[2:4], value[0:2])

                # fix values
                if value[5:7] == '02' and value[8:10] == '30':  # using >28 for February
                    value = '28'.join(value.rsplit('30', 1))
        return value

    def apply_dtype(x: Any) -> Any:
        """
        Handles `default` `datetime.datetime` for `dateutil.parser.parse`
        """
        if dtype == parser.parse:
            return dtype(x, default=default_datetime)
        return dtype(x)

    # apply the rules and data type change
    dataframe[col_name] = dataframe[col_name].apply(
        lambda x: apply_dtype(standardize(x)) if x is not None else func(x))

    return dataframe


@loggingdecorator(logger.name+'.func', level=logging.INFO, output=False)
def dump_directory_structure_csv(src: str, shallow: bool = True) -> None:
    """
    Takes a `src` directory path, creates a tree of dir structure and writes
        it down to a csv file with name 'label.csv' with
        default value of '0' for each path

    args:
        src: Source directory path
        shallow: If only dive one level of depth (False: recursive)
    """

    dic = create_directory_structure_tree(src=src, shallow=shallow)
    flat_dic = flatten_dict(dic)
    flat_dic = {k: v for k, v in flat_dic.items() if v is not None}
    dict_to_csv(d=flat_dic, path=src+'/label.csv')


def create_directory_structure_tree(src: str, shallow: bool = False) -> dict:
    """
    Takes a path to directory and creates a dictionary of it its tree directory structure

    ref: https://stackoverflow.com/a/25226267/18971263
    args:
        src: Path to source directory
        shallow: Whether or not just dive to root dir's subdir
    """
    d = {'name': os.path.basename(src) if os.path.isdir(
        src) else None}  # ignore files, only dir
    if os.path.isdir(src):
        if shallow:
            d['children'] = [{x: '0'} for x in os.listdir(src)]  # type: ignore
        else:  # recursively walk into all dirs and subdirs
            d['children'] = [create_directory_structure_tree(  # type: ignore
                os.path.join(src, x)) for x in os.listdir(src)]
    else:
        pass
        # d['type'] = "file"
    return d


def flatten_dict(d: dict) -> dict:
    """
    Takes a (nested) multilevel dictionary and flattens it where the final keys are key.key....
        and values are the leaf values of dictionary.

    ref: https://stackoverflow.com/a/67744709/18971263
    args:
        d: A dictionary  
    """

    def items():
        if isinstance(d, dict):
            for key, value in d.items():
                # nested subtree
                if isinstance(value, dict):
                    for subkey, subvalue in flatten_dict(value).items():
                        yield '{}.{}'.format(key, subkey), subvalue
                # nested list
                elif isinstance(value, list):
                    for num, elem in enumerate(value):
                        for subkey, subvalue in flatten_dict(elem).items():
                            yield '{}.[{}].{}'.format(key, num, subkey), subvalue
                # everything else (only leafs should remain)
                else:
                    yield key, value
    return dict(items())

def xml_to_flattened_dict(xml: str) -> dict:
        """
        Takes a (nested) XML and flattens it to a dict where the final keys are key.key....
            and values are the leaf values of XML tree.

        args:
            xml: A XML string
        """
        flattened_dict = xmltodict.parse(xml)  # XML to dict
        flattened_dict = flatten_dict(flattened_dict)
        return flattened_dict


def unit_converter(sparse: float, dense: float, factor: float) -> float:
    """
    convert `sparse` or `dense` to each other using
        the rule of thump of `dense = (factor) sparse` or `sparse = (1./factor) dense`.

    args:
        sparse: the smaller/sparser amount which is a percentage of `dense`,\n
            if provided calculates `sparse = (factor) dense`.
        dense: the larger/denser amount which is a multiplication of `sparse`,\n
            if provided calculates `dense = (1/factor) sparse`
        factor: sparse to dense factor, either directly provided as a\n
            float number or as a predefined factor given by `constant.FINANCIAL_RATIOS`
    """
    # only sparse or dense must exist
    assert not (sparse is not None and dense is not None)

    if sparse is not None:
        dense = factor * sparse
        return dense
    if dense is not None:
        sparse = (1./factor) * dense
        return sparse

# methods used for handling files from manually processed dataset to raw-dataset
#   see class `FileTransformers` in `preprocessor.py` for more information


def process_directory(src_dir: str, dst_dir: str, compose: FileTransformCompose,
                      file_pattern: str = '*'):
    """
    Iterates through `src_dir`, processing all files that match pattern, the applies
        given transformation composition `compose` and stores them,
        including their parent directories in `dst_dir`.

    args:
        src_dir: Source directory to be processed
        dst_dir: Destination directory to write processed files
        file_pattern: pattern to match files, default to '*' for all files
        compose: An instance of transform composer, see `preprocessor.Compose`

    ref: https://stackoverflow.com/a/24041933/18971263
    """

    assert src_dir != dst_dir, 'Source and destination dir must differ.'
    if src_dir[-1] != '/':
        src_dir += '/'
    for dirpath, dirnames, all_filenames in os.walk(src_dir):
        # filter out files that match pattern only
        filenames = filter(lambda fname: fnmatch(
            fname, file_pattern), all_filenames)

        if filenames:
            dir_ = os.path.join(dst_dir, dirpath.replace(src_dir, ''))
            os.makedirs(dir_, exist_ok=True)
            for fname in filenames:
                in_fname = os.path.join(dirpath, fname)
                out_fname = os.path.join(dir_, fname)
                compose(in_fname, out_fname)


@loggingdecorator(logger.name+'.func', level=logging.DEBUG, output=True, input=True)
def search_dict(string: str, dic: dict, if_nan: str) -> str:
    """
    Converts the (custom and non-standard) code of a country to its name given the XFA docs LOV section.
    args:
        string: input code string
        dic: dictionary to be searched throw the keys for `string`
        if_nan: if `string` could not be found in `dic`, return `if_nan`
    """
    country = [c for c in dic.keys() if string in c]
    if country:
        return dic[country[0]]
    else:
        logger.debug('"{}" key could not be found, filled with "{}".'.format(string, if_nan))
        return if_nan

@loggingdecorator(logger.name+'.func', level=logging.DEBUG, output=False, input=True)
def config_csv_to_dict(path: str) -> dict:
    """
    Takes a config CSV and return a dictionary of key and values

    args:
        path: string path to config file
    """

    config_df = pd.read_csv(path)
    return dict(zip(config_df[config_df.columns[0]], config_df[config_df.columns[1]]))
