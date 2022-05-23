"""
Only run this code for generating [dataset_name].pkl file, every time we want to create a new version
    of our dataset, then after this step, we use DVC to version it.
In main.py, we just rerun this step but integrated into MLFlow to track which version we are
    USING (than generating).

In simple terms, if you added new samples, changed columns or anything that should be considered
    permanent at the time, you should run this script, then version it with DVC and for doing
    data analysis or machine learning for prediction, only pull from DVC remote storage of 
    this version (or any version you want).
"""

from utils.constant import DOC_TYPES
from utils import functional
from utils.preprocessor import *

import os
import pandas as pd


# main path
SRC_DIR = '/mnt/e/dataset/processed/all/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/all/'  # path to decrypted pdf

# main code
# Canada protected PDF to machine readable for all entries and transferring other files as it is
compose = {
    CopyFile(mode='cf'): '.csv',
    CopyFile(mode='cf'): '.txt',
    MakeContentCopyProtectedMachineReadable(): '.pdf'
}
file_transform_compose = FileTransformCompose(transforms=compose)
functional.process_directory(src_dir=SRC_DIR, dst_dir=DST_DIR,
                             compose=file_transform_compose, file_pattern='*')

SRC_DIR = DST_DIR[:-1]
dataframe = pd.DataFrame()
for dirpath, dirnames, all_filenames in os.walk(SRC_DIR):
    dataframe_entry = pd.DataFrame()

    # filter all_filenames
    filenames = all_filenames
    if filenames:
        files = [os.path.join(dirpath, fname) for fname in filenames]
        # applicant info
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

# save dataframe to disc as pickle
dataset_path = DST_DIR[:-1] + '.pkl'
dataframe.to_pickle(dataset_path)
