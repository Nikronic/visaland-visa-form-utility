from enum import Enum


# dict of abbreviation used to shortening length of KEYS in XML to CSV conversion
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
    'BackgroundInfo': 'BGI',
    'Current': 'Curr',
    'Previous': 'Prev',
    'Marriage': 'Marr',
    'Married': 'Marr',
    'Previously': 'Prev',
    'Passport': 'Psprt',
    'Language': 'Lang',
    'Address': 'Addr',
    'contact': 'cntct',
    'Contact': 'cntct',
    'Resident': 'Resi',
    'Phone': 'Phn',
    'Number': 'Num',
    'Purpose': 'Prps',
    'HowLongStay': 'HLS',
    'Signature': 'Sign',
}

CANADA_5645E_KEY_ABBREVIATION = {
    'page': 'p',
    'Applicant': 'App',
    'Mother': 'Mo',
    'Father': 'Fa',
    'Section': 'Sec',
    'Spouse': 'Sps',
    'Child': 'Chd',
}

# dict of abbreviation used to shortening length of VALUES in XML to CSV conversion
CANADA_5257E_VALUE_ABBREVIATION = {
    'BIOMETRIC ENROLMENT': 'Bio'
}


class DOC_TYPES(Enum):
    """
    Contains all document types which can be used to customize ETL steps for each doc.
    Remark: Order of docs is meaningless.
    """
    
    canada_5257e = 1  # application for visitor visa (temporary resident visa)
    canada_5645e = 2  # Family information

# Cutoff terms for both keys and values
class CANADA_CUTOFF_TERMS(Enum):
    """
    Dict of cut off terms for different files that is can be used with
        'dict_summarizer' for `functional`
    """

    ca5645e = 'IMM_5645'
    ca5257e = 'form1'
