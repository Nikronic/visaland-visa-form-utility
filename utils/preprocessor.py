import pandas as pd
import numpy as np

import re
from dateutil import parser


T0 = '19000202T000000'


def column_dropper(dataframe: pd.DataFrame, string: str, regex: bool = False) -> None:
    """
    Takes a Pandas Dataframe and searches for columns *containing* `string` in them either 
        raw string or regex (in latter case, use `regex=True`) and drops them *in-place*.

    args:
        dataframe: Pandas dataframe to be processed
        string: string to look for in `dataframe` columns
        regex: compile `string` as regex

    """

    if regex:
        r = re.compile(string)
        col_to_drop = list(filter(r.match, dataframe.columns.values))
    else:
        col_to_drop = [
            col for col in dataframe.columns.values if string in col]

    dataframe.drop(col_to_drop, axis=1, inplace=True)


def fillna_datetime(dataframe: pd.DataFrame, col_base_name: str, one_sided: str = False,
                    date: str = None) -> None:
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
    """

    if one_sided is not None:
        r = re.compile(col_base_name.replace('.', '\.'))
    else:    
        r = re.compile(col_base_name.replace('.', '\.')+'\.(From|To).+')
    columns_to_fillna_names = list(filter(r.match, dataframe.columns.values))
    if date is None:
        dataframe[columns_to_fillna_names] = dataframe[columns_to_fillna_names].fillna(
            T0, inplace=False)
    else:
        dataframe[columns_to_fillna_names] = dataframe[columns_to_fillna_names].fillna(
            date, inplace=False)


def aggregate_datetime(dataframe: pd.DataFrame, col_base_name: str, new_col_name: str,
                       one_sided: str = None, reference_date: str = None, current_date: str = None) -> pd.DataFrame:
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
    """

    aggregated_column_name = None
    if one_sided is None:
        aggregated_column_name = col_base_name + '.' + new_col_name
        r = re.compile(col_base_name.replace('.', '\.')+'\.(From|To).+')
    else:  # when one_sided, we no longer have *From* or *To*
        aggregated_column_name = col_base_name + '.' + new_col_name
        r = re.compile(col_base_name.replace('.', '\.'))
    columns_to_aggregate_names = list(
        filter(r.match, dataframe.columns.values))
    dataframe[aggregated_column_name] = np.nan  # combination of dates

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

    if isinstance(column_from_date, str):
        column_from_date = parser.parse(column_from_date)
    if isinstance(column_to_date, str):
        column_to_date = parser.parse(column_to_date)

    if column_from_date is None:  # ignore reference_date if from_date exists
        dataframe[from_date] = dataframe[from_date].apply(parser.parse)
        column_from_date = dataframe[from_date]
    if column_to_date is None:  # ignore current_date if to_date exists
        dataframe[to_date] = dataframe[to_date].apply(parser.parse)
        column_to_date = dataframe[to_date]
    dataframe[aggregated_column_name].fillna(
        column_to_date - column_from_date, inplace=True)  # period
    dataframe[aggregated_column_name] = dataframe[aggregated_column_name].dt.days.astype(
        'int32')  # change to int of days

    dataframe.drop(columns_to_aggregate_names, axis=1,
                   inplace=True)  # drop from/to columns
    return dataframe
