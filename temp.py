import os
import shutil
import logging
import sys
import enlighten

from vizard_data.constant import DOC_TYPES
from vizard_data import functional
from vizard_data.preprocessor import *

import dvc.api
import pandas as pd


# configure logging
VERBOSITY = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(VERBOSITY)
logger_handler = logging.StreamHandler(sys.stderr)
logger.addHandler(logger_handler)
# set libs to log to our logging config
__libs = ['vizard_data', 'vizard_models', 'vizard_snorkel']
for __l in __libs:
    __libs_logger = logging.getLogger(__l)
    __libs_logger.setLevel(logging.INFO)
    __libs_logger.addHandler(logger_handler)

manager = enlighten.get_manager(sys.stderr)  # setup progress bar

logger.info(
    '\t\t↓↓↓ Starting setting up configs: dirs, dvc, etc ↓↓↓')
# main path
SRC_DIR = '/mnt/e/dataset/processed/all/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/all/'  # path to decrypted pdf


# data versioning config
PATH = DST_DIR[:-1] + '.pkl'  # path to source data, e.g. data.pkl file
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.0.2.1'

logger.info('DVC data version: {}'.format(VERSION))
logger.info('DVC repo (root): {}'.format(REPO))
logger.info('DVC data source path: {}'.format(PATH))
logger.info(
    '\t\t↑↑↑ Finished setting up configs: dirs, dvc, etc ↑↑↑')

# main code
logger.info('\t\t↓↓↓ Starting data extraction ↓↓↓')
# Canada protected PDF to machine readable for all entries and transfering other files as it is
compose = {
    CopyFile(mode='cf'): '.csv',
    CopyFile(mode='cf'): '.txt',
    MakeContentCopyProtectedMachineReadable(): '.pdf'
}
file_transform_compose = FileTransformCompose(transforms=compose)
functional.process_directory(src_dir=SRC_DIR, dst_dir=DST_DIR,
                             compose=file_transform_compose, file_pattern='*')
logger.info('\t\t↑↑↑ Finished data extraction ↑↑↑')

logger.info('\t\t↓↓↓ Starting data preprocessing ↓↓↓')
# convert PDFs to pandas dataframes
data_iter_logger = logging.getLogger(logger.name+'.data_iter')

SRC_DIR = DST_DIR[:-1]
dataframe = pd.DataFrame()
progress_bar = manager.counter(total=len(next(os.walk(DST_DIR), (None, [], None))[1]),
                               desc='Preprocessed', unit='data point')
i = 0  # for progress bar
for dirpath, dirnames, all_filenames in os.walk(SRC_DIR):
    dataframe_entry = pd.DataFrame()

    # filter all_filenames
    filenames = all_filenames
    if filenames:
        files = [os.path.join(dirpath, fname) for fname in filenames]
        # applicant form
        in_fname = [f for f in files if '5257' in f][0]
        df_preprocessor = CanadaDataframePreprocessor()
        if len(in_fname) != 0:
            dataframe_applicant = df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_5257e)
        # applicant family info
        in_fname = [f for f in files if '5645' in f][0]
        if len(in_fname) != 0:
            dataframe_family = df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_5645e)
        # manually added labels
        in_fname = [f for f in files if 'label' in f][0]
        if len(in_fname) != 0:
            dataframe_label = df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_label)

        # final dataframe: concatenate common forms and label column wise
        dataframe_entry = pd.concat(
            objs=[dataframe_applicant, dataframe_family, dataframe_label],
            axis=1, verify_integrity=True)

    # concat the dataframe_entry into the main dataframe (i.e. adding rows)
    dataframe = pd.concat(objs=[dataframe, dataframe_entry], axis=0,
                          verify_integrity=True, ignore_index=True)
    # logging
    i += 1
    data_iter_logger.info('Processed {}th data point...'.format(i))
    progress_bar.update()


# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
# read dataset from remote (local) data storage
dataframe_original = pd.read_pickle(data_url)
logger.info('\t\t↑↑↑ Finished data preprocessing ↑↑↑')

import pandas.testing as pdt
for c in dataframe_original.columns.values:
    try:
        pdt.assert_series_equal(left=dataframe[c], right=dataframe_original[c],
                                check_exact=False, rtol=1e-3, atol=1e-4)
    except AssertionError as e:
        print(f'column="{c}" {e}\n\n')

