"""
Only run this code for generating [dataset_name].pkl file, every time we want to create a new version
    of our dataset, then after this step, we use DVC to version it.
In main.py, we just retrun this step but integrated into MLFlow to track which version we are
    USING (than generating).

In simple terms, if you added new samples, changed columns or anything that should be considered
    permanent at the time, you should run this script, then version it with DVC and for doing
    data analysis or machine learning for prediction, only pull from DVC remote storage of 
    this version (or any version you want).
"""

from utils.PDFIO import CanadaXFA
from utils.preprocessor import CanadaDataframePreprocessor
from utils.constant import DOC_TYPES

import os
import pandas as pd


# main path
SRC_DIR = '/mnt/e/dataset/processed/h0/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/h0/'  # path to decrypted pdf

# main code
# Canada protected PDF to machine readable for all entries
canada_xfa = CanadaXFA()
canada_xfa.process_directory(src_dir=SRC_DIR, dst_dir=DST_DIR, pattern='*.pdf',
                             func=canada_xfa.make_machine_readable)

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
