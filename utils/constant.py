from enum import Enum


# dict of abbreviation used to shortening length of keys in XML to CSV conversion
CANADA_5257E_KEY_ABBREVIATION = {
    'Page': 'P',
    'PersonalDetails': 'PD',
    'CountryWhereApplying': 'CWA',
    'MaritalStatus': 'MS',
    'Section': 'Sec',
    'ContactInformation': 'CI',
    'DetailsOfVisit': 'DOV',
    'Education': 'Edu',
    'PageWrapper': 'PW',
    'Occupation': 'Occ',
    'BackgroundInfo': 'BGI'
}
CANADA_5645E_KEY_ABBREVIATION = {
    'age': 'p',
    'Applicant': 'App',
    'Mother': 'Mo',
    'Father': 'Fa',
    'Section': 'Sec',
}

class DOC_TYPES(Enum):
    """
    Contains all document types which can be used to customize ETL steps for each doc.
    Remark: Order of docs is meaningless.
    """
    
    canada_5257e = 1  # application for visitor visa (temporary resident visa)
    canada_5645e = 2  # Family information

# 
class CANADA_CUTOFF_TERMS(Enum):
    """
    Dict of cut off terms for different files that is can be used with
        'dict_summarizer' for `functional`
    """

    ca5645e = 'IMM_5645'
    ca5257e = 'form1'
