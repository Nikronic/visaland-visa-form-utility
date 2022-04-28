from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils import functional
from utils import CANADA_5257E_KEY_ABBREVIATION, CANADA_5257E_VALUE_ABBREVIATION
from utils import CANADA_CUTOFF_TERMS

src_path = 'sample/enc_sample/'  # path to source encrypted pdf
des_path = 'sample/dec_sample/'  # path to decrypted pdf
SAMPLE = 'sample/dec_sample/imm5257e.pdf'

# Canada PDF to XML
canada_xfa = CanadaXFA()
canada_xfa.make_machine_readable(
    file_name=None, src_path=src_path, des_path=des_path)
xml = canada_xfa.extract_raw_content(SAMPLE)

# XFA to XML
xml = canada_xfa.clean_xml_for_csv(xml=xml, type=DOC_TYPES.canada_5257e)
# XML to flattened dict
data_dict = canada_xfa.xml_to_flattened_dict(xml=xml)
data_dict = canada_xfa.flatten_dict(data_dict)
# clean flattened dict
data_dict = functional.dict_summarizer(data_dict, cutoff_term=CANADA_CUTOFF_TERMS.ca5257e.value,
                                       KEY_ABBREVIATION_DICT=CANADA_5257E_KEY_ABBREVIATION,
                                       VALUE_ABBREVIATION_DICT=CANADA_5257E_VALUE_ABBREVIATION)
# write to csv
functional.dict_to_csv(d=data_dict, path='sample/csv/5257e.csv')

