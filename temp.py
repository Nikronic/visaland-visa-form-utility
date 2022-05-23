from typing import Union
import os
import shutil
import logging
import uuid
import enlighten

from vizard_utils.constant import *
from vizard_utils import functional
from vizard_utils.preprocessor import *

import pandas as pd

# main path
FILE_NAME = 'databank-2015-2019.csv'
SRC_DIR = 'raw-dataset/field_data/'
SRC_FILE = SRC_DIR + FILE_NAME

# main code
dataframe = pd.read_csv(SRC_FILE)

# save dataframe to disc as pickle
dataset_path = 'raw-dataset/' + FILE_NAME[:-4] + '.pkl'
dataframe.to_pickle(dataset_path)
