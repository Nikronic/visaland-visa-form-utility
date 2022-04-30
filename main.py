from dateutil import parser
import re
from utils.PDFIO import CanadaXFA
from utils.constant import DOC_TYPES
from utils import functional
from utils.preprocessor import *
from utils import CANADA_5257E_KEY_ABBREVIATION, CANADA_5257E_VALUE_ABBREVIATION
from utils import CANADA_5257E_DROP_COLUMNS
from utils import CANADA_CUTOFF_TERMS

import pandas as pd

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
# functional.dict_to_csv(d=data_dict, path='sample/csv/5257e.csv')

# convert each data dict to a dataframe
dataframe = pd.DataFrame.from_dict(data=[data_dict], orient='columns')

# add rows to main dataframe
dataframe2 = pd.concat(objs=[dataframe, dataframe], axis=0)

# drop pepeg columns
dataframe.drop(CANADA_5257E_DROP_COLUMNS, axis=1, inplace=True)

# transform multiple pleb columns into a single chad one (e.g. *.FromDate and *.ToDate --> *.Period)
# *.FromDate and *.ToDate --> *.Period


# column dtypes
print
# age to integer
dataframe['P1.Age'] = dataframe['P1.Age'].astype('int16')
# Adult binary state: adult=True or child=False
dataframe['P1.AdultFlag'] = dataframe['P1.AdultFlag'].apply(
    lambda x: True if x == 'adult' else False)
# service language: 1=En, 2=Fr -> need to be changed to categorical
dataframe['P1.PD.ServiceIn.ServiceIn'] = dataframe['P1.PD.ServiceIn.ServiceIn'].apply(
    lambda x: 'En' if x == 1 else 'Fr')
# AliasNameIndicator: 1=True, 0=False
dataframe['P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator'] = dataframe['P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator'].apply(
    lambda x: True if x == 'Y' else False)
# VisaType: String, We may need to remove it if its all the same everywhere, otherwise categorical
dataframe['P1.PD.VisaType.VisaType'] = dataframe['P1.PD.VisaType.VisaType'].astype(
    'string')
# Birth City: String -> categorical
dataframe['P1.PD.PlaceBirthCity'] = dataframe['P1.PD.PlaceBirthCity'].astype(
    'string')
# Birth country: string -> categorical
dataframe['P1.PD.PlaceBirthCountry'] = dataframe['P1.PD.PlaceBirthCountry'].astype(
    'string')
# citizen of: string -> categorical
dataframe['P1.PD.Citizenship.Citizenship'] = dataframe['P1.PD.Citizenship.Citizenship'].astype(
    'string')
# current country of residency: string -> categorical
dataframe['P1.PD.CurrCOR.Row2.Country'] = dataframe['P1.PD.CurrCOR.Row2.Country'].astype(
    'string')
# current country of residency status: string -> categorical
dataframe['P1.PD.CurrCOR.Row2.Status'] = dataframe['P1.PD.CurrCOR.Row2.Status'].astype(
    'string')
# current country of residency other descritpion: bool -> categorical
dataframe['P1.PD.CurrCOR.Row2.Other'] = dataframe['P1.PD.CurrCOR.Row2.Other'].apply(
    lambda x: True if x is not None else False)
# date of birth in year: int days
dataframe['P1.PD.DOBYear'] = dataframe['P1.PD.DOBYear'].apply(parser.parse)
# validation date of information, i.e. current date: datetime
dataframe['P3.Sign.C1CertificateIssueDate'] = dataframe['P3.Sign.C1CertificateIssueDate'].apply(
    parser.parse)
# current country of residency period: Datetime -> int days
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P1.PD.CurrCOR.Row2',
                               new_col_name='Period', reference_date=dataframe['P1.PD.DOBYear'],
                               current_date=dataframe['P3.Sign.C1CertificateIssueDate'])
# delete tnx to P1.PD.CurrCOR.Row2
column_dropper(dataframe=dataframe, string='P1.PD.CORDates')
# has previous country of residency: bool -> categorical
dataframe['P1.PD.PCRIndicator'] = dataframe['P1.PD.PCRIndicator'].apply(
    lambda x: True if x == 'Y' else False)
# previous country of residency 02: string (na=0)-> categorical
dataframe['P1.PD.PrevCOR.Row2.Country'] = dataframe['P1.PD.PrevCOR.Row2.Country'].astype(
    'string').fillna('0')
# previous country of residency status 02: string (na=0)-> categorical
dataframe['P1.PD.PrevCOR.Row2.Status'] = dataframe['P1.PD.PrevCOR.Row2.Status'].astype(
    'string').fillna('0')
# previous country of residency 02 period (P1.PD.PrevCOR.Row2): none -> random date -> int days
fillna_datetime(dataframe=dataframe, col_base_name='P1.PD.PrevCOR.Row2')
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P1.PD.PrevCOR.Row2',
                               new_col_name='Period', reference_date=None,
                               current_date=None)
# previous country of residency 03: string (na=0)-> categorical
dataframe['P1.PD.PrevCOR.Row3.Country'] = dataframe['P1.PD.PrevCOR.Row3.Country'].astype(
    'string').fillna('0')
# previous country of residency status 03: string (na=0)-> categorical
dataframe['P1.PD.PrevCOR.Row3.Status'] = dataframe['P1.PD.PrevCOR.Row3.Status'].astype(
    'string').fillna('0')
# previous country of residency 03 period (P1.PD.PrevCOR.Row3): none -> random date -> int days
fillna_datetime(dataframe=dataframe, col_base_name='P1.PD.PrevCOR.Row3')
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P1.PD.PrevCOR.Row3',
                               new_col_name='Period', reference_date=None,
                               current_date=None)
# delete tnx to P1.PD.PrevCOR.Row2 and P1.PD.PrevCOR.Row3
column_dropper(dataframe=dataframe, string='P1.PD.PCRDatesR')
# apply from country of residency (cwa=country where apply): Y=True, N=False
dataframe['P1.PD.SameAsCORIndicator'] = dataframe['P1.PD.SameAsCORIndicator'].apply(
    lambda x: True if x == 'Y' else False)
# country where applying: string -> categorical
dataframe['P1.PD.CWA.Row2.Country'] = dataframe['P1.PD.CWA.Row2.Country'].astype(
    'string')
# country where applying status: string -> categorical
dataframe['P1.PD.CWA.Row2.Status'] = dataframe['P1.PD.CWA.Row2.Status'].astype(
    'string')
# country where applying other: string -> categorical, maybe delete entirely
dataframe['P1.PD.CWA.Row2.Other'] = dataframe['P1.PD.CWA.Row2.Other'].astype(
    'string')
# country where applying period: datetime -> int days
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P1.PD.CWA.Row2',
                               new_col_name='Period')
# delete tnx to P1.PD.CWA.Row2
column_dropper(dataframe=dataframe, string='P1.PD.CWADates')
# marriage period: datetime -> int days
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P1.MS.SecA.DateOfMarr',
                               one_sided='right', new_col_name='Period', reference_date=None,
                               current_date=dataframe['P3.Sign.C1CertificateIssueDate'])
# delete tnx to P1.MS.SecA.DateOfMarr
column_dropper(dataframe=dataframe, string='P1.MS.SecA.MarrDate.From')
# previous marriage: Y=True, N=False
dataframe['P2.MS.SecA.PrevMarrIndicator'] = dataframe['P2.MS.SecA.PrevMarrIndicator'].apply(
    lambda x: True if x == 'Y' else False)

# FIXME: doesn't fill value at all and when it does, it is NaT rather than correct `date``
# previous spouse age: none -> random date -> int days
# fillna_datetime(dataframe=dataframe, date=dataframe['P3.Sign.C1CertificateIssueDate'],
#                 col_base_name='P2.MS.SecA.PrevSpouseDOB.DOBYear', one_sided=True)
# dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P2.MS.SecA.PrevSpouseDOB.DOBYear',
#                                new_col_name='Period', reference_date=None, one_sided='right',
#                                current_date=dataframe['P3.Sign.C1CertificateIssueDate'], )

# previous marriage period: none -> random date -> int days
fillna_datetime(dataframe=dataframe, col_base_name='P2.MS.SecA')
dataframe = aggregate_datetime(dataframe=dataframe, col_base_name='P2.MS.SecA',
                               new_col_name='Period', reference_date=None,
                               current_date=None)
# delete tnx to P2.MS.SecA.FromDate and P2.MS.SecA.ToDate.ToDate
column_dropper(dataframe=dataframe, string='P2.MS.SecA.Prevly')
