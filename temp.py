from dateutil import parser
import re
from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils.functional import *
from utils.preprocessor import *
from utils import *

import pandas as pd

src_path = 'sample/enc_sample/'  # path to source encrypted pdf
des_path = 'sample/dec_sample/'  # path to decrypted pdf

# Canada PDF to XML
canada_xfa = CanadaXFA()
canada_xfa.make_machine_readable(
    file_name=None, src_path=src_path, des_path=des_path)

SAMPLE = 'sample/dec_sample/imm5645e.pdf'
xml = canada_xfa.extract_raw_content(SAMPLE)

# XFA to XML
xml = canada_xfa.clean_xml_for_csv(xml=xml, type=DOC_TYPES.canada_5645e)
# XML to flattened dict
data_dict = canada_xfa.xml_to_flattened_dict(xml=xml)
data_dict = canada_xfa.flatten_dict(data_dict)
# clean flattened dict
data_dict = functional.dict_summarizer(data_dict, cutoff_term=CANADA_CUTOFF_TERMS.ca5645e.value,
                                       KEY_ABBREVIATION_DICT=CANADA_5645E_KEY_ABBREVIATION,
                                       VALUE_ABBREVIATION_DICT=None)
# write to csv
# functional.dict_to_csv(d=data_dict, path='sample/csv/5645e.csv')

# convert each data dict to a dataframe
dataframe = pd.DataFrame.from_dict(data=[data_dict], orient='columns')

# add rows to main dataframe
dataframe2 = pd.concat(objs=[dataframe, dataframe], axis=0)

# drop pepeg columns
# 5645e Canada is way easier to programmitically delete columns, hence we avoid hardcoding
dataframe.drop(CANADA_5645E_DROP_COLUMNS, axis=1, inplace=True)

# transform multiple pleb columns into a single chad one (e.g. *.FromDate and *.ToDate --> *.Period)
# *.FromDate and *.ToDate --> *.Period
# column dtypes

# type of application -> onehot (str) -> onehot (int16)
cols = [col for col in dataframe.columns.values if 'p1.Subform1' in col]
dataframe[cols] = dataframe[cols].astype(np.int16)
# drop all names
functional.column_dropper(dataframe=dataframe, string='Name', inplace=True)
# drop all addresses
functional.column_dropper(dataframe=dataframe, string='Addr', inplace=True)
# drop all Accompany=No and only rely on Accompany=Yes using binary state
functional.column_dropper(dataframe=dataframe, string='No', inplace=True)
# drop tail of section (it's too much to consider 3rd sibling of someone's job/DOB/etc)
# TODO: convert this into another variable that says someone "has foriegner sibling"
#   also can convert it into "sibling living in foriegn country"
print

# applictant marriage status: string to integer
dataframe['p1.SecA.App.ChdMStatus'] = dataframe['p1.SecA.App.ChdMStatus'].astype(
    'int16')
# spouse date of birth: string -> datetime
dataframe['p1.SecA.Sps.SpsDOB'] = dataframe['p1.SecA.Sps.SpsDOB'].apply(
    parser.parse)
# validation date of information, i.e. current date: datetime
dataframe['p1.SecC.SecCdate'] = dataframe['p1.SecC.SecCdate'].apply(
    parser.parse)
# spouse age period: datetime -> int days
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='p1.SecA.Sps.SpsDOB',
                               type=DOC_TYPES.canada, new_col_name='Period',
                               reference_date=dataframe['p1.SecA.Sps.SpsDOB'],
                               current_date=dataframe['p1.SecC.SecCdate'], one_sided='right')
# spouse country of birth: string -> categorical
dataframe['p1.SecA.Sps.SpsCOB'] = dataframe['p1.SecA.Sps.SpsCOB'].astype(
    'string')
# spouse occupation type (issue #1, #2, #3): string, employee, student, housewife, entrepreneur, etc -> categorical
dataframe['p1.SecA.Sps.SpsOcc'] = dataframe['p1.SecA.Sps.SpsOcc'].astype(
    'string')
# spouse accompanying: coming=True or not_coming=False
dataframe['p1.SecA.Sps.SpsAccomp'] = dataframe['p1.SecA.Sps.SpsAccomp'].apply(
    lambda x: False if x == '0' else True)
# mother date of birth: string -> datetime
dataframe['p1.SecA.Mo.MoDOB'] = dataframe['p1.SecA.Mo.MoDOB'].apply(
    parser.parse)
# mother age period: datetime -> int days
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='p1.SecA.Mo.MoDOB',
                               type=DOC_TYPES.canada, new_col_name='Period',
                               reference_date=dataframe['p1.SecA.Mo.MoDOB'],
                               current_date=dataframe['p1.SecC.SecCdate'], one_sided='right')
# mother occupation type (issue #1, #2, #3): string, employee, student, housewife, entrepreneur, etc -> categorical
dataframe['p1.SecA.Mo.MoOcc'] = dataframe['p1.SecA.Mo.MoOcc'].astype('string')
# mother accompanying: coming=True or not_coming=False
dataframe['p1.SecA.Mo.MoAccomp'] = dataframe['p1.SecA.Mo.MoAccomp'].apply(
    lambda x: False if x == '0' else True)
# father date of birth: string -> datetime
dataframe['p1.SecA.Fa.FaDOB'] = dataframe['p1.SecA.Fa.FaDOB'].apply(
    parser.parse)
# father age period: datetime -> int days
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='p1.SecA.Fa.FaDOB',
                               type=DOC_TYPES.canada, new_col_name='Period',
                               reference_date=dataframe['p1.SecA.Fa.FaDOB'],
                               current_date=dataframe['p1.SecC.SecCdate'], one_sided='right')
# mother occupation type (issue #1, #2, #3): string, employee, student, housewife, entrepreneur, etc -> categorical
dataframe['p1.SecA.Fa.FaOcc'] = dataframe['p1.SecA.Fa.FaOcc'].astype('string')
# father accompanying: coming=True or not_coming=False
dataframe['p1.SecA.Fa.FaAccomp'] = dataframe['p1.SecA.Fa.FaAccomp'].apply(
    lambda x: False if x == '0' else True)

# children's status
children_tag_list = [c for c in dataframe.columns.values if 'p1.SecB.Chd' in c]
CHILDREN_MAX_FEATURES = 6
for i in range(len(children_tag_list) // CHILDREN_MAX_FEATURES):
    # child's marriage status 01: string to integer
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecB.Chd.['+str(i)+'].ChdMStatus',
                             dtype=np.int8, if_nan='skip')
    # child's relationship 01: string -> categorical
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecB.Chd.['+str(i)+'].ChdRel',
                             dtype=str, if_nan='skip')

    # child's date of birth 01: string -> datetime
    # fill nan's with average age of children
    # if no date available: fill with average difference age of kids from their parents
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecB.Chd.['+str(i)+'].ChdDOB',
                             dtype=parser.parse, if_nan='skip')
    # child's age period 01: datetime -> int days
    dataframe = aggregate_datetime(dataframe=dataframe, type=DOC_TYPES.canada,
                                   col_base_name='p1.SecB.Chd.[' +
                                   str(i)+'].ChdDOB', new_col_name='Period', 
                                   reference_date=None, one_sided='right',
                                   current_date=dataframe['p1.SecC.SecCdate'],
                                   if_nan='skip')
    # child's country of birth 01: string -> categorical
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecB.Chd.['+str(i)+'].ChdCOB',
                             dtype=str, if_nan='fill', value='IRAN')
    # child's occupation type 01 (issue #1, #2, #3): string, employee, student, housewife, entrepreneur, etc -> categorical
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecB.Chd.['+str(i)+'].ChdOcc',
                             dtype=str, if_nan='skip')
    # child's accompanying 01: coming=True or not_coming=False
    dataframe['p1.SecB.Chd.['+str(i)+'].ChdAccomp'] = dataframe['p1.SecB.Chd.['+str(i)+'].ChdAccomp'].apply(
        lambda x: False if x == '0' else True)

# siblings' status
siblings_tag_list = [c for c in dataframe.columns.values if 'p1.SecC.Chd' in c]
SIBLINGS_MAX_FEATURES = 6
for i in range(len(siblings_tag_list) // CHILDREN_MAX_FEATURES):
    # sibling's marriage status 01: string to integer
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecC.Chd.['+str(i)+'].ChdMStatus',
                             dtype=np.int8, if_nan='skip')
    # sibling's relationship 01: string -> categorical
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecC.Chd.['+str(i)+'].ChdRel',
                             dtype=str, if_nan='skip')
    # sibling's date of birth 01: string -> datetime
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecC.Chd.['+str(i)+'].ChdDOB',
                             dtype=parser.parse, if_nan='skip')
    # sibling's age period 01: datetime -> int days
    dataframe = aggregate_datetime(dataframe=dataframe, type=DOC_TYPES.canada,
                                   col_base_name='p1.SecC.Chd.[' +
                                   str(i)+'].ChdDOB', new_col_name='Period', 
                                   reference_date=None, one_sided='right',
                                   current_date=dataframe['p1.SecC.SecCdate'],
                                   if_nan='skip')
    # sibling's country of birth 01: string -> categorical
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecC.Chd.['+str(i)+'].ChdCOB',
                             dtype=str, if_nan='fill', value='IRAN')
    # sibling's occupation type 01 (issue #1, #2, #3): string, employee, student, housewife, entrepreneur, etc -> categorical
    dataframe = change_dtype(dataframe=dataframe, col_name='p1.SecC.Chd.['+str(i)+'].ChdOcc',
                             dtype=str, if_nan='skip')
    # sibling's accompanying: coming=True or not_coming=False
    dataframe['p1.SecC.Chd.['+str(i)+'].ChdAccomp'] = dataframe['p1.SecC.Chd.['+str(i)+'].ChdAccomp'].apply(
        lambda x: False if x == '0' else True)
print
