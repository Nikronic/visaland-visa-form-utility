import dvc.api
from mlflow import log_metric, log_param, log_artifacts
import mlflow
import os
from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.preprocessor import CanadaDataframePreprocessor

import pandas as pd


SRC_DIR = '/mnt/e/dataset/raw/10-5-2022-h/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/10-5-2022-h/'  # path to decrypted pdf

# Canada PDF to XML for all files
canada_xfa = CanadaXFA()
canada_xfa.process_directory(src_dir=SRC_DIR, dst_dir=DST_DIR,
                             func=canada_xfa.make_machine_readable)

# create dataframes for each common form
# SAMPLE_APPLICANT = 'sample/dec_sample/2/imm5257e-MOHAMMAD.pdf'
# SAMPLE_FAMILY = 'sample/dec_sample/2/imm5645e-MOHAMMAD.pdf'
# SAMPLE_APPLICANT = 'sample/dec_sample/1/imm5257e.pdf'
# SAMPLE_FAMILY = 'sample/dec_sample/1/imm5645e.pdf'
# SAMPLE_APPLICANT = 'sample/dec_sample/3/imm5257e-FARAHNAZ.pdf'
# SAMPLE_FAMILY = 'sample/dec_sample/3/imm5645e (2)-FARAHNAZ.pdf'

# ### concat all data, finalize dataframe
# TODO see if on the fly preprocessing is faster or offline one


# decrypted pdfs as source dir for obtaining pandas dataframe
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
        if len(in_fname) != 0:  # applicant form
            ca5257e_df_preprocessor = CanadaDataframePreprocessor()
            dataframe_applicant = ca5257e_df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_5257e)
        in_fname = [f for f in files if '5645' in f][0]
        if len(in_fname) != 0:
            ca5645e_df_preprocessor = CanadaDataframePreprocessor()
            dataframe_family = ca5645e_df_preprocessor.file_specific_basic_transform(
                path=in_fname, type=DOC_TYPES.canada_5645e)

        # concatenate common forms column wise
        dataframe_entry = pd.concat(
            objs=[dataframe_applicant, dataframe_family], axis=1, verify_integrity=True)

    # concat the dataframe_entry into the main dataframe (i.e. adding rows)
    dataframe = pd.concat(objs=[dataframe, dataframe_entry], axis=0,
                          verify_integrity=True, ignore_index=True)
print

# ### MLFlow


# data versioning config
# PATH = DST_DIR[:-1] + '.pkl'  # path to source data, e.g. data.pkl file
# REPO = '/home/nik/visaland-visa-form-utility'
# VERSION = 


# log_param("param1", randint(0, 100))
# # Log a metric; metrics can be updated throughout the run
# log_metric("foo", random())
# log_metric("foo", random() + 1)
# log_metric("foo", random() + 2)

# # Log an artifact (output file)
# output_path = "outputs"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# with open(output_path+"/test.txt", "w") as f:
#     f.write("hello world!")
# log_artifacts(output_path)


# # ### MLFlow
# print
