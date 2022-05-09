import os
from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.preprocessor import CanadaDataframePreprocessor

import pandas as pd


src_dir = 'sample/enc_sample/'  # path to source encrypted pdf
dst_dir = 'sample/dec_sample/'  # path to decrypted pdf

# Canada PDF to XML for all files
canada_xfa = CanadaXFA()
canada_xfa.process_directory(src_dir=src_dir, dst_dir=dst_dir,
                             func=canada_xfa.make_machine_readable)

src_dir = 'sample/dec_sample'
dataframe = pd.DataFrame()
for dirpath, dirnames, all_filenames in os.walk(src_dir):
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
