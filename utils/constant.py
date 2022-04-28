from enum import Enum

## DICTIONARY
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

    # more meaningful keys
    'GovPosition.Choice': 'witnessIllTreat',
    'Occ.Choice': 'politicViol',
    'BGI3.Choice': 'criminalRec',
    'Details.VisaChoice3': 'PrevApply',
    'BGI2.VisaChoice2': 'refuseDeport',
    'BGI2.VisaChoice1': 'noAuthStay',
    'backgroundInfoCalc': 'otherThanMedic'
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

## LIST
CANADA_5257E_DROP_COLUMNS = [
    'ns0:datasets.@xmlns:ns0', 'P1.Header.CRCNum', 'P1.FormVersion', 'P1.PrevSpouseAge',
    'P1.PD.UCIClientID', 'P1.PD.SecHeader.@ns0:dataNode', 'P1.PD.DOBYear', 'P1.PD.DOBMonth',
    'P1.PD.DOBDay', 'P1.PD.CurrCOR.Row1.@ns0:dataNode', 'P1.PD.PrevCOR.Row1.@ns0:dataNode', 
    'P1.PD.CWA.Row1.@ns0:dataNode', 'P1.PD.ApplicationValidatedFlag', 
    'P2.MS.SecA.SecHeader.@ns0:dataNode', 'P2.MS.SecA.DateLastValidated.DateCalc', 
    'P2.MS.SecA.DateLastValidated.Year', 'P2.MS.SecA.DateLastValidated.Month', 
    'P2.MS.SecA.DateLastValidated.Day', 'P2.MS.SecA.PsprtSecHeader.@ns0:dataNode', 
    'P2.MS.SecA.Psprt.IssueYYYY', 'P2.MS.SecA.Psprt.IssueMM', 'P2.MS.SecA.Psprt.IssueDD',
    'P2.MS.SecA.Psprt.expiryYYYY', 'P2.MS.SecA.Psprt.expiryMM', 'P2.MS.SecA.Psprt.expiryDD',
    'P2.MS.SecA.Langs.languagesHeader.@ns0:dataNode', 'P2.natID.SecHeader.@ns0:dataNode',
    'P2.natID.natIDdocs.IssueDate.IssueDate', 'P2.natID.natIDdocs.ExpiryDate',
    'P2.USCard.SecHeader.@ns0:dataNode', 'P2.USCard.SecHeader.@ns0:dataNode', 
    'P2.USCard.usCarddocs.ExpiryDate', 'P2.USCard.usCarddocs.DocNum.DocNum', 
    'P2.CI.cntct.cntctInfoSecHeader.@ns0:dataNode', 'P2.CI.cntct.AddrRow1.POBox.POBox',
    'P2.CI.cntct.AddrRow1.Apt.AptUnit', 'P2.CI.cntct.AddrRow1.StreetNum.StreetNum', 
    'P2.CI.cntct.AddrRow1.Streetname.Streetname', 'P2.CI.cntct.AddrRow2.ProvinceState.ProvinceState',
    'P2.CI.cntct.AddrRow2.PostalCode.PostalCode', 'P2.CI.cntct.AddrRow2.District',
    'P2.CI.cntct.ResiialAddrRow1.AptUnit.AptUnit', 'P2.CI.cntct.ResiialAddrRow1.StreetNum.StreetNum',
    'P2.CI.cntct.ResiialAddrRow1.StreetName.Streetname', 'P2.CI.cntct.ResiialAddrRow2.District',
    'P2.CI.cntct.ResiialAddrRow2.ProvinceState.ProvinceState', 'P2.CI.cntct.PhnNums.Phn.NumCountry',
    'P2.CI.cntct.ResiialAddrRow2.PostalCode.PostalCode', 'P2.CI.cntct.PhnNums.Phn.ActualNum', 
    'P2.CI.cntct.PhnNums.Phn.NANum.AreaCode', 'P2.CI.cntct.PhnNums.Phn.NANum.FirstThree', 
    'P2.CI.cntct.PhnNums.Phn.NANum.LastFive', 'P2.CI.cntct.PhnNums.Phn.IntlNum.IntlNum', 
    'P2.CI.cntct.PhnNums.AltPhn.NumCountry', 'P2.CI.cntct.PhnNums.AltPhn.ActualNum',
    'P2.CI.cntct.PhnNums.AltPhn.NANum.AreaCode', 'P2.CI.cntct.PhnNums.AltPhn.NANum.FirstThree',
    'P2.CI.cntct.PhnNums.AltPhn.NANum.LastFive', 'P2.CI.cntct.PhnNums.AltPhn.IntlNum.IntlNum',
    'P2.CI.cntct.FaxEmail.Phn.CanadaUS', 'P2.CI.cntct.FaxEmail.Phn.Other', 
    'P2.CI.cntct.FaxEmail.Phn.NumExt',
    'P2.CI.cntct.FaxEmail.Phn.NumCountry', 'P2.CI.cntct.FaxEmail.Phn.ActualNum', 
    'P2.CI.cntct.FaxEmail.Phn.NANum.AreaCode', 'P2.CI.cntct.FaxEmail.Phn.NANum.FirstThree',
    'P2.CI.cntct.FaxEmail.Phn.NANum.LastFive', 'P2.CI.cntct.FaxEmail.Phn.IntlNum.IntlNum',
    'P2.CI.cntct.FaxEmail.Email', 'P3.SecHeader_DOV.@ns0:dataNode', 
    'P3.DOV.cntcts_Row1.AddrInCanada.AddrInCanada', 'P3.Edu.Edu_SecHeader.@ns0:dataNode',
    'P3.Occ.SecHeader_CurrOcc.@ns0:dataNode', 'P3.BGI_SecHeader.@ns0:dataNode',
    'P3.Sign.Consent0.Choice', 'P3.Sign.hand.@ns0:dataNode', 'P3.Sign.TextField2',
    'P3.Disclosure.@ns0:dataNode', 'P3.ReaderInfo', 'Barcodes.@ns0:dataNode', 
    'P3.BGI2.Details.refusedDetails', 'P3.PWrapper.BGI3.details',
    'P3.PWrapper.Military.militaryServiceDetails',
]

## ENUM
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
