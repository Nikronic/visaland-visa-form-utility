import os
import shutil
import logging
import uuid
import enlighten

from vizard_utils.constant import DOC_TYPES
from vizard_utils import functional
from vizard_utils.preprocessor import *

import dvc.api
import mlflow
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
SRC_DIR = '/mnt/e/dataset/processed/all/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/all/'  # path to decrypted pdf

# MLFlow configs
# data versioning config
PATH = DST_DIR[:-1] + '.pkl'  # path to source data, e.g. data.pkl file
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v0.0.2'

# log experiment configs
MLFLOW_EXPERIMENT_NAME = 'setting up logging integrated into mlflow'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_TAGS = {
    'stage': 'dev'  # dev, beta, production
}
mlflow.set_tags(MLFLOW_TAGS)

logger.info('MLflow experiment name: {}'.format(MLFLOW_EXPERIMENT_NAME))
logger.info('MLflow experiment id: {}'.format(mlflow.active_run().info.run_id))
logger.info('MLflow data version: {}'.format(VERSION))
logger.info('MLflow repo (root): {}'.format(REPO))
logger.info('MLflow data source path: {}'.format(PATH))
logger.info(
    '\t\t↑↑↑ Finished setting up configs: dirs, mlflow, dvc, etc ↑↑↑')

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

# convert PDFs to pandas dataframes
data_iter_logger = logging.getLogger(logger.name+'.data_iter')

SRC_DIR = DST_DIR[:-1]
dataframe = pd.DataFrame()
progress_bar = manager.counter(total=len(next(os.walk(DST_DIR), (None, [], None))[1]),
                               desc='Ticks', unit='ticks')
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
    data_iter_logger.info('Processed {}th data point ...'.format(i))
    progress_bar.update()

# save dataframe to disc as pickle
logger.info('\t\t↓↓↓ Starting saving dataframe to disc ↓↓↓')
dataset_path = DST_DIR[:-1] + '.pkl'
dataframe.to_pickle(dataset_path)
logger.info('Dataframe saved to path={}'.format(dataset_path))
logger.info('\t\t↑↑↑ Finished saving dataframe to disc ↑↑↑')

# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)

# read dataset from remote (local) data storage
dataframe = pd.read_pickle(data_url)
logger.info('\t\t↑↑↑ Finished data extraction ↑↑↑')

# log data params
logger.info('\t\t↓↓↓ Starting logging with MLFlow ↓↓↓')

mlflow.log_param('data_url', data_url)
mlflow.log_param('raw_dataset_dir', DST_DIR)
mlflow.log_param('data_version', VERSION)
mlflow.log_param('input_shape', dataframe.shape)
mlflow.log_param('input_columns', dataframe.columns.values)
mlflow.log_param('input_dtypes', dataframe.dtypes.values)

logger.info('\t\t↑↑↑ Finished logging with MLFlow ↑↑↑')

# Log artifacts (logs, saved files, etc)
mlflow.log_artifacts('artifacts/')
# delete redundant logs, files that are logged as artifact
shutil.rmtree('artifacts')
