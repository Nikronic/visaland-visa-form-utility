from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.preprocessor import CanadaDataframePreprocessor
from utils import *

import pandas as pd


src_path = 'sample/enc_sample/'  # path to source encrypted pdf
des_path = 'sample/dec_sample/'  # path to decrypted pdf

# Canada PDF to XML for all files
canada_xfa = CanadaXFA()
canada_xfa.make_machine_readable(
    file_name=None, src_path=src_path, des_path=des_path)

# create dataframes for each common form
SAMPLE_APPLICANT = 'sample/dec_sample/imm5257e.pdf'
SAMPLE_FAMILY = 'sample/dec_sample/imm5645e.pdf'
ca5257e_df_preprocessor = CanadaDataframePreprocessor()
dataframe_applicant = ca5257e_df_preprocessor.file_specific_basic_transform(
    path=SAMPLE_APPLICANT, type=DOC_TYPES.canada_5257e)
ca5645e_df_preprocessor = CanadaDataframePreprocessor()
dataframe_family = ca5645e_df_preprocessor.file_specific_basic_transform(
    path=SAMPLE_FAMILY, type=DOC_TYPES.canada_5645e)

# concatenate common forms column wise
dataframe = pd.concat(
    objs=[dataframe_applicant, dataframe_family], axis=1, verify_integrity=True)
