from typing import Union
import os
import shutil
import logging
import uuid
from xml.dom.pulldom import IGNORABLE_WHITESPACE
import enlighten

from vizard_utils.constant import *
from vizard_utils import functional
from vizard_utils.preprocessor import *

import pandas as pd

# main path
FILE_NAME = 'API_NY.GDP.PCAP.CD_DS2_en_xml_v2_4004943.xml'
SRC_DIR = 'raw-dataset/field_data/'
SRC_FILE = SRC_DIR + FILE_NAME

# main code
dataframe = pd.read_xml(
    SRC_FILE, xpath='/Root/record/field', elems_only=False, )

# save dataframe to disc as pickle
dataset_path = 'raw-dataset/' + FILE_NAME[:-4] + '.pkl'
dataframe.to_pickle(dataset_path)

# process df
dataframe = dataframe.pivot(columns='name', values='field')
dataframe = dataframe.drop('Item', axis=1)
dataframe['Country or Area'] = dataframe['Country or Area'].ffill().bfill()
dataframe = dataframe.drop_duplicates()
dataframe = dataframe.ffill().bfill()
dataframe = dataframe[dataframe['Year'].astype(int) >= 2017]
dataframe = dataframe.drop_duplicates(subset=['Country or Area', 'Year'], keep='last').reset_index()
df2 = dataframe.pivot(index='index', columns='Year', values='Value')
dataframe.drop('index', axis=1, inplace=True)
dataframe.reset_index(inplace=True)
df2.reset_index(inplace=True)
dataframe = df2.join(dataframe['Country or Area'])

country_names = dataframe['Country or Area'].unique()
for cn in country_names:
    dataframe[dataframe['Country or Area'] ==
              cn] = dataframe[dataframe['Country or Area'] == cn].ffill().bfill()

# dataframe = dataframe.ffill().bfill()
dataframe = dataframe.drop_duplicates(subset=['Country or Area'])
dataframe.drop('index', axis=1, inplace=True)
mean_columns = [c for c in dataframe.columns.values if c.isnumeric()]
dataframe['mean'] = dataframe[mean_columns].astype(float).mean(axis=1)
dataframe.drop(dataframe.columns[:-2], axis=1, inplace=True)

dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].apply(
    lambda x: x.lower())

# scale to [1-7] (Standard of World Data Bank)
column_max = dataframe['mean'].max()
column_min = dataframe['mean'].min()

def standardize(x):
    return (((x - column_min) * (7. - 1.)) /
                (column_max - column_min)) + 1.
dataframe['mean'] = dataframe['mean'].apply(standardize)
dic = dict(zip(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]]))

# # save dataframe to disc as pickle
# dataset_path = 'raw-dataset/' + FILE_NAME[:-4] + '.pkl'
# dataframe.to_pickle(dataset_path)
