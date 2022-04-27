from enum import Enum


# dict of abbreviation used to shortening length of keys in XML to CSV conversion
CANADA_5257E_KEY_ABBREVIATION = {
    'Page1': 'P1',
    'PersonalDetails': 'PD',
    'CountryWhereApplying': 'CWA',
    'MaritalStatus': 'MS',
    'SectionA': '',
    'ContactInformation': 'CI',
    'DetailsOfVisit': 'DOV',
    'Education': 'Edu',
    'PageWrapper': 'PW',
    'Occupation': 'Occ',
    'BackgroundInfo': 'BGI'
}

class DOC_TYPES(Enum):
    """
    Contains all document types which can be used to customize ETL steps for each doc.
    Remark: Order of docs is meaningless.
    """
    
    canada_5257e = 1  # application for visitor visa (temporary resident visa)
    canada_5645e = 2  # Family information

