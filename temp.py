from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.preprocessor import CanadaDataframePreprocessor
from utils import *

import pandas as pd


src_dir = 'sample/enc_sample/'  # path to source encrypted pdf
dst_dir = 'sample/dec_sample/'  # path to decrypted pdf

# Canada PDF to XML for all files
canada_xfa = CanadaXFA()
canada_xfa.process_directory(src_dir=src_dir, dst_dir=dst_dir,
                             func=canada_xfa.make_machine_readable)

# create dataframes for each common form
# SAMPLE_APPLICANT = 'sample/dec_sample/2/imm5257e-MOHAMMAD.pdf'
# SAMPLE_FAMILY = 'sample/dec_sample/2/imm5645e-MOHAMMAD.pdf'
# SAMPLE_APPLICANT = 'sample/dec_sample/1/imm5257e.pdf'
# SAMPLE_FAMILY = 'sample/dec_sample/1/imm5645e.pdf'
SAMPLE_APPLICANT = 'sample/dec_sample/3/imm5257e-FARAHNAZ.pdf'
SAMPLE_FAMILY = 'sample/dec_sample/3/imm5645e (2)-FARAHNAZ.pdf'

ca5257e_df_preprocessor = CanadaDataframePreprocessor()
dataframe_applicant = ca5257e_df_preprocessor.file_specific_basic_transform(
    path=SAMPLE_APPLICANT, type=DOC_TYPES.canada_5257e)
ca5645e_df_preprocessor = CanadaDataframePreprocessor()
dataframe_family = ca5645e_df_preprocessor.file_specific_basic_transform(
    path=SAMPLE_FAMILY, type=DOC_TYPES.canada_5645e)

# concatenate common forms column wise
dataframe = pd.concat(
    objs=[dataframe_applicant, dataframe_family], axis=1, verify_integrity=True)
print
