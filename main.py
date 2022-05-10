import os
from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.preprocessor import CanadaDataframePreprocessor

import dvc.api
import mlflow
import pandas as pd


SRC_DIR = '/mnt/e/dataset/processed/10-5-2022-h/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/10-5-2022-h/'  # path to decrypted pdf

# MLFlow configs
# data versioning config
PATH = DST_DIR[:-1] + '.pkl'  # path to source data, e.g. data.pkl file
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v0.0.1'

# log experiment configs
MLFLOW_EXPERIMENT_NAME = 'real case dataset data extraction'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_TAGS = {
    'stage': 'dev'  # dev, beta, production
}
mlflow.set_tags(MLFLOW_TAGS)


# main code
# Canada protected PDF to machine readable for all entries
canada_xfa = CanadaXFA()
canada_xfa.process_directory(src_dir=SRC_DIR, dst_dir=DST_DIR, pattern='*.pdf',
                             func=canada_xfa.make_machine_readable)

# convert PDFs to pandas dataframes
SRC_DIR = DST_DIR[:-1]
dataframe = pd.DataFrame()
for dirpath, dirnames, all_filenames in os.walk(SRC_DIR):
    dataframe_entry = pd.DataFrame()

    # filter all_filenames
    filenames = all_filenames
    if filenames:
        files = [os.path.join(dirpath, fname) for fname in filenames]
        # process files
        # take automated forms
        in_fname = [f for f in files if '5257' in f][0]
        df_preprocessor = CanadaDataframePreprocessor()
        if len(in_fname) != 0:  # applicant form
            dataframe_applicant = df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_5257e)
        in_fname = [f for f in files if '5645' in f][0]
        if len(in_fname) != 0:
            dataframe_family = df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_5645e)

        # concatenate common forms column wise
        dataframe_entry = pd.concat(
            objs=[dataframe_applicant, dataframe_family], axis=1, verify_integrity=True)

    # concat the dataframe_entry into the main dataframe (i.e. adding rows)
    dataframe = pd.concat(objs=[dataframe, dataframe_entry], axis=0,
                          verify_integrity=True, ignore_index=True)

# save dataframe to disc as pickle
dataset_path = DST_DIR[:-1] + '.pkl'
dataframe.to_pickle(dataset_path)


# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)

# read dataset from remote (local) data storage 
dataframe = pd.read_pickle(data_url)

# log data params
mlflow.log_param('data_url', data_url)
mlflow.log_param('raw_dataset_dir', DST_DIR)
mlflow.log_param('data_version', VERSION)
mlflow.log_param('input_shape', dataframe.shape)
mlflow.log_param('input_columns', dataframe.columns.values)
mlflow.log_param('input_dtypes', dataframe.dtypes.values)
