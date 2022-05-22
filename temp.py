from tracemalloc import start
from typing import Union
import os
import shutil
import logging
import uuid
import enlighten

from utils.constant import DOC_TYPES
from utils import functional
from utils.preprocessor import *

import pandas as pd


# configure logging
VERBOSITY = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(VERBOSITY)
# Set up root logger, and add a file handler to root logger
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')
    os.makedirs('artifacts/logs')

log_file_name = uuid.uuid4()
logger_handler = logging.FileHandler(filename='artifacts/logs/{}.log'.format(log_file_name),
                                     mode='w')
logger.addHandler(logger_handler)
manager = enlighten.get_manager()  # setup progress bar

logger.info(
    '\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')
# main path
FILE_NAME = 'databank-2015-2019.csv'
SRC_DIR = 'raw-dataset/field_data/'
SRC_FILE = SRC_DIR + FILE_NAME

# main code
logger.info('\t\t↓↓↓ Starting loading raw data ↓↓↓')
dataframe = pd.read_csv(SRC_FILE)
logger.info('\t\t↑↑↑ Finished loading raw data ↑↑↑')



# constans
INDICATOR = 'Indicator'
SUBINDICATOR = 'Subindicator Type'
subindicator_rank = False
SUBINDICATOR_TYPE = 'Rank' if subindicator_rank else '1-7 Best'

years = (2015, 2017)

# drop useless columns
columns_to_drop = ['Country ISO3', 'Indicator Id', ]
columns_to_drop = columns_to_drop + \
    [c for c in dataframe.columns.values if '-' in c]
dataframe.drop(columns_to_drop, axis=1, inplace=True)

# figure out start and end year index of columns names values
start_year, end_year = [
    str(y) for y in years] if years is not None else (None, None)
column_years = [c for c in dataframe.columns.values if c.isnumeric()]
start_year_index = column_years.index(
    start_year) if start_year is not None else 0
end_year_index = column_years.index(
    end_year) if end_year is not None else -1
# dataframe with desired years
sub_column_years = column_years[start_year_index: end_year_index+1]
columns_to_drop = [c for c in list(
    set(column_years) - set(sub_column_years)) if c.isnumeric()]
dataframe.drop(columns_to_drop, axis=True, inplace=True)

# filter rows that only contain the provided `indicator_name` with type `rank` or `score`
indicator_name = 'Quality of the education system, 1-7 (best)'
dataframe = dataframe[(dataframe[INDICATOR] == indicator_name) &
                        (dataframe[SUBINDICATOR] == SUBINDICATOR_TYPE)]
dataframe.drop([INDICATOR, SUBINDICATOR], axis=1, inplace=True)

dataframe[indicator_name + '_mean'] = dataframe.mean(axis=1, skipna=True,
                                                     numeric_only=True)





# save dataframe to disc as pickle
logger.info('\t\t↓↓↓ Starting saving dataframe to disc ↓↓↓')
dataset_path = SRC_DIR[:-1] + FILE_NAME[:-4] + '.pkl'
dataframe.to_pickle(dataset_path)
logger.info('Dataframe saved to path={}'.format(dataset_path))
logger.info('\t\t↑↑↑ Finished saving dataframe to disc ↑↑↑')
