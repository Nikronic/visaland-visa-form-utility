from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.preprocessor import CanadaDataframePreprocessor
from utils import *


src_path = 'sample/enc_sample/'  # path to source encrypted pdf
des_path = 'sample/dec_sample/'  # path to decrypted pdf

# ######
# ALL FILES
# ######
# Canada PDF to XML for all files
canada_xfa = CanadaXFA()
canada_xfa.make_machine_readable(
    file_name=None, src_path=src_path, des_path=des_path)

# ######
# 5257e
# ######
# file specific loading
SAMPLE = 'sample/dec_sample/imm5257e.pdf'
ca_df_preprocessor = CanadaDataframePreprocessor()
dataframe = ca_df_preprocessor.file_specific_basic_transform(
    path=SAMPLE, type=DOC_TYPES.canada_5257e)
