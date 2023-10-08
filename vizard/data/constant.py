__all__ = [
    'CANADA_5257E_KEY_ABBREVIATION', 'CANADA_5645E_KEY_ABBREVIATION', 'CANADA_5257E_VALUE_ABBREVIATION',
    'CANADA_5257E_DROP_COLUMNS', 'CANADA_5645E_DROP_COLUMNS', 'FINANCIAL_RATIOS', 'DOC_TYPES',
    'CANADA_CUTOFF_TERMS', 'CANADA_FILLNA', 'DATEUTIL_DEFAULT_DATETIME', 'FEATURE_NAME_TO_TEXT_MAP',

    # Data Enums shared all over the place
    'CustomNamingEnum', 'CanadaMarriageStatus',
    'CanadaContactRelation', 'CanadaResidencyStatus', 'Sex', 'EducationFieldOfStudy',
    'CountryWhereApplying', 'PurposeOfVisit', 'OccupationTitle',

    # Constants (similar to Enums but not inherited from Enum)
    'CanadaGeneralConstants'
]

import datetime
from enum import Enum, auto
from types import DynamicClassAttribute
from typing import List


# DICTIONARY
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
"""Dict of abbreviation used to shortening length of KEYS in XML to CSV conversion

"""

CANADA_5645E_KEY_ABBREVIATION = {
    'page': 'p',
    'Applicant': 'App',
    'Mother': 'Mo',
    'Father': 'Fa',
    'Section': 'Sec',
    'Spouse': 'Sps',
    'Child': 'Chd',
    'Address': 'Addr',
    'Occupation': 'Occ',
    'Yes': 'Accomp',
    'Relationship': 'Rel',

}
"""Dict of abbreviation used to shortening length of KEYS in XML to CSV conversion

"""

# see #29
DATEUTIL_DEFAULT_DATETIME = {
    'day': 18,  # no reason for this value (CluelessClown)
    'month': 6,  # no reason for this value (CluelessClown)
    'year': datetime.MINYEAR
}
"""A default date for the ``dateutil.parser.parse`` function when some part of date is not provided

"""

CANADA_5257E_VALUE_ABBREVIATION = {
    'BIOMETRIC ENROLMENT': 'Bio',
    '223': 'IRAN',
    '045': 'TURKEY',
}
"""Dict of abbreviation used to shortening length of VALUES in XML to CSV conversion
"""

# LIST
CANADA_5257E_DROP_COLUMNS = [
    'ns0:datasets.@xmlns:ns0', 'P1.Header.CRCNum', 'P1.FormVersion', 'P1.PrevSpouseAge',
    'P1.PD.UCIClientID', 'P1.PD.SecHeader.@ns0:dataNode', 'P1.PD.DOBMonth',
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
    'P2.CI.cntct.FaxEmail.Phn.NumExt', 'P2.MS.SecA.Psprt.IssueDate.IssueDate',
    'P2.CI.cntct.FaxEmail.Phn.NumCountry', 'P2.CI.cntct.FaxEmail.Phn.ActualNum',
    'P2.CI.cntct.FaxEmail.Phn.NANum.AreaCode', 'P2.CI.cntct.FaxEmail.Phn.NANum.FirstThree',
    'P2.CI.cntct.FaxEmail.Phn.NANum.LastFive', 'P2.CI.cntct.FaxEmail.Phn.IntlNum.IntlNum',
    'P2.CI.cntct.FaxEmail.Email', 'P3.SecHeader_DOV.@ns0:dataNode',
    'P3.DOV.cntcts_Row1.AddrInCanada.AddrInCanada', 'P3.Edu.Edu_SecHeader.@ns0:dataNode',
    'P3.Occ.SecHeader_CurrOcc.@ns0:dataNode', 'P3.BGI_SecHeader.@ns0:dataNode',
    'P3.Sign.Consent0.Choice', 'P3.Sign.hand.@ns0:dataNode', 'P3.Sign.TextField2',
    'P3.Disclosure.@ns0:dataNode', 'P3.ReaderInfo', 'Barcodes.@ns0:dataNode',
    'P3.BGI2.Details.refusedDetails', 'P3.PWrapper.BGI3.details',
    'P3.PWrapper.Military.militaryServiceDetails', 'P1.PD.AliasName.AliasGivenName',
    'P1.PD.AliasName.AliasFamilyName', 'P2.MS.SecA.PMFamilyName', 'P2.MS.SecA.GivenName.PMGivenName',
    'P3.DOV.cntcts_Row1.Name.Name', 'P3.cntcts_Row2.Name.Name', 'P1.PD.PrevCOR.Row3.Other',
    'P3.cntcts_Row2.AddrInCanada.AddrInCanada', 'P3.Edu.Edu_Row1.FromMonth', 'P3.Edu.Edu_Row1.ToMonth',
    'P3.Edu.Edu_Row1.School', 'P3.Edu.Edu_Row1.CityTown', 'P3.Edu.Edu_Row1.ProvState',
    'P3.Occ.OccRow1.FromMonth', 'P3.Occ.OccRow1.ToMonth', 'P3.Occ.OccRow1.CityTown.CityTown',
    'P3.Occ.OccRow1.ProvState', 'P3.Occ.OccRow2.FromMonth', 'P3.Occ.OccRow2.ToMonth',
    'P3.Occ.OccRow2.CityTown.CityTown', 'P3.Occ.OccRow2.ProvState', 'P3.Occ.OccRow3.FromMonth',
    'P3.Occ.OccRow3.ToMonth', 'P3.Occ.OccRow3.CityTown.CityTown', 'P3.Occ.OccRow3.ProvState',
    'P1.PD.Name.FamilyName', 'P1.PD.Name.GivenName', 'P1.PD.PrevCOR.Row2.Other',
    'P1.MS.SecA.FamilyName', 'P1.MS.SecA.GivenName', 'P2.MS.SecA.PrevSpouseDOB.DOBMonth',
    'P2.MS.SecA.PrevSpouseDOB.DOBDay', 'P2.MS.SecA.Psprt.PsprtNum.PsprtNum',
    'P2.natID.natIDdocs.DocNum.DocNum', 'P2.MS.SecA.Langs.languages.lov', 'P3.Occ.OccRow1.Employer',
    'P3.Occ.OccRow2.Employer', 'P3.Occ.OccRow3.Employer', 'P1.Age',
    'P2.MS.SecA.Psprt.TaiwanPIN', 'P2.MS.SecA.Psprt.IsraelPsprtIndicator',
]
"""List of columns to be dropped before doing any preprocessing

Note:
    This list has been determined manually.

"""

CANADA_5645E_DROP_COLUMNS = {
    'xfa:datasets.@xmlns:xfa', 'p1.SecA.Title.@xfa:dataNode', 'p1.SecA.App.AppDOB', 'p1.SecA.App.AppCOB',
    'p1.SecA.App.AppOcc', 'p1.SecA.App.AppOcc', 'p1.SecB.SecBsignature',
    'p1.SecB.SecBdate', 'p1.SecC.Title.@xfa:dataNode', 'p1.SecA.SecAsignature',
    'p1.SecA.SecAdate', 'p1.SecB.Title.@xfa:dataNode', 'p1.SecC.SecCsignature',
    'p1.SecC.Subform2.@xfa:dataNode', 'p1.SecA.Sps.ChdMStatus', 'formNum',
}
"""List of columns to be dropped before doing any preprocessing

Note:
    This list has been determined manually.

"""

FINANCIAL_RATIOS = {
    # house related
    'rent2deposit': 100./3.,  # rule of thumb provided online
    'deposit2rent': 3./100.,
    'deposit2worth': 5.,      # rule of thumb provided online
    'worth2deposit': 1./5.,
    # company related
    'tax2income': 100./15.,  # 10% for small, 20% for larger, we use average 15% tax rate
    'income2tax': 15./100.,
    # a company worth 15x of its income for minimum: C[T/E]O suggestion
    'income2worth': 15.,
    'worth2income': 1./15.,
}
"""Ratios used to convert rent, deposit, and total worth to each other

Note:
    This is part of dictionaries containing factors in used in heuristic
    calculations using domain knowledge

"""


# ENUM
class DOC_TYPES(Enum):
    """Contains all document types which can be used to customize ETL steps for each document type

    Members follow the ``<country_name>_<document_type>`` naming convention. The value 
    and its order are meaningless.
    """
    canada = 1        # referring to all Canada docs in general
    canada_5257e = 2  # application for visitor visa (temporary resident visa)
    canada_5645e = 3  # Family information
    canada_label = 4  # containing labels


# Cutoff terms for both keys and values
class CANADA_CUTOFF_TERMS(Enum):
    """Dict of cut off terms for different files that is can be used with :func:`dict_summarizer <vizard.data.functional.dict_summarizer>`

    """

    ca5645e = 'IMM_5645'
    ca5257e = 'form1'


# values used to fill None's depending on the form structure
#   remark: we do not use any heuristics here, we just follow what form used and only add another
# option which should be used as None state (i.e. None as a separate feature in categorical mode).
class CANADA_FILLNA(Enum):
    """Values used to fill ``None`` s depending on the form structure

    Members follow the ``<field_name>_<form_name>`` naming convention. The value
    has been extracted by manually inspecting the documents. Hence, for each
    form, user must find and set this value manually.

    Note:
        We do not use any heuristics here, we just follow what form used and
        only add another option which should be used as ``None`` state; i.e. ``None``
        as a separate feature in categorical mode.
    """

    ChdMStatus_5645e = 9


class CustomNamingEnum(Enum):
    """Extends base :class:`enum.Enum` to support custom naming for members

    Note:
        Class attribute :attr:`name` has been overridden to return the name
        of a marital status that matches with the dataset and not the ``Enum``
        naming convention of Python. For instance, ``COMMON_LAW`` -> ``common-law`` in
        case of Canada forms.

    Note:
        Devs should subclass this class and add their desired members in newly
        created classes. E.g. see :class:`CanadaMarriageStatus`

    Note:
        Classes that subclass this, for values of their members should use :class:`enum.auto`
        to demonstrate that chosen value is not domain-specific. Otherwise, any explicit
        value given to members should implicate a domain-specific (e.g. extracted from dataset)
        value. Values that are explicitly provided are the values used in original data. Hence,
        it should not be modified by any means as it is tied to dataset, transformation,
        and other domain-specific values. E.g. compare values in :class:`CanadaMarriageStatus`
        and :class:`SiblingRelation`.
    """

    @DynamicClassAttribute
    def name(self):
        _name = super(CustomNamingEnum, self).name
        _name: str = _name.lower()
        # convert FOO_BAR to foo-bar (dataset convention)
        _name = _name.replace('_', '-')
        self._name_ = _name
        return self._name_

    @classmethod
    def get_member_names(cls):
        _member_names_: List[str] = []
        for mem_name in cls._member_names_:
            _member_names_.append(cls._member_map_[mem_name].name)
        return _member_names_


class CanadaMarriageStatus(CustomNamingEnum):
    """States of marriage in Canada forms

    Note:
        Values for the members are the values used in original Canada forms. Hence,
        it should not be modified by any means as it is tied to dataset, transformation,
        and other domain-specific values.
    """

    COMMON_LAW = 2
    DIVORCED = 3
    SEPARATED = 4
    MARRIED = 5
    SINGLE = 7
    WIDOWED = 8
    UNKNOWN = 9


class CanadaContactRelation(CustomNamingEnum):
    """Contact relation in Canada data
    """

    F1 = auto()
    F2 = auto()
    HOTEL = auto()
    WORK = auto()
    FRIEND = auto()
    UKN = auto()


class CanadaResidencyStatus(CustomNamingEnum):
    """Residency status in a country in Canada data
    """

    CITIZEN = 1
    VISITOR = 3
    OTHER = 6


class EducationFieldOfStudy(CustomNamingEnum):
    """Field of study types in general
    """

    APPRENTICE = auto()
    DIPLOMA = auto()
    BACHELOR = auto()
    MASTER = auto()
    PHD = auto()
    UNEDU = auto()

class OccupationTitle(CustomNamingEnum):
    """Occupation title (position) types in general

    Todo:
        ``HOUSEWIFE`` need to be deleted and assumed ``'OTHER'`` or something similar.
    """

    MANAGER = auto()
    STUDENT = auto()
    RETIRED = auto()
    SPECIALIST = auto()
    EMPLOYEE = auto()
    HOUSEWIFE = auto()
    OTHER = auto()

    # TODO: OTHER name has to be be ``'OTHER'``, fix the below hardcoding
    @DynamicClassAttribute
    def name(self):
        _name = super(CustomNamingEnum, self).name
        _name: str = _name.lower()
        # convert FOO_BAR to foo-bar (dataset convention)
        _name = _name.replace('_', '-')
        # set `OTHER`'s name to 'OTHER'
        if _name == 'other':
            _name = 'OTHER'
        self._name_ = _name
        return self._name_


class Sex(CustomNamingEnum):
    """Sex types in general
    """

    FEMALE = auto()
    MALE = auto()

    @DynamicClassAttribute
    def name(self):
        _name = super(CustomNamingEnum, self).name
        # convert foobar to Foobar (i.e. Female, Male)
        _name: str = _name.lower().capitalize()
        self._name_ = _name
        return self._name_


class CountryWhereApplying(CustomNamingEnum):
    """Countries that applicant can go for fingerprint (i.e., apply for visa from)
    """

    ARMENIA = auto()
    GEORGIA = auto()
    UAE     = auto()
    TURKEY  = auto()
    OTHER   = auto()

    @DynamicClassAttribute
    def name(self):
        _name = super(CustomNamingEnum, self).name
        _name: str = _name.upper()
        self._name_ = _name
        return self._name_


class PurposeOfVisit(CustomNamingEnum):
    """The purpose of visit for requested visa
    """

    FAMILY_VISIT = auto()
    VISIT        = auto()
    TOURISM      = auto()
    OTHER        = auto()

    @DynamicClassAttribute
    def name(self):
        _name = super(CustomNamingEnum, self).name
        _name: str = _name.lower()
        if _name == 'family_visit':
            return 'family visit'
        self._name_ = _name
        return self._name_


class CanadaGeneralConstants:
    """Constant values such as thresholds and defaults in general case

    Note: 
        Since members are not necessarily related to each other, 
        the naming and the value suggests the use case. This is just to prevent
        using hardcoded values and also creating too many types/classes without 
        much other uses cases other than being a placeholder for constant values.
    """
    MINIMUM_AGE: float = 18.
    """Minimum age for ``date_of_birth``
    """

    MINIMUM_EXPIRY_PASSPORT: float = 0.5
    """Minimum time required for `passport_expiry_date_remaining`
    """
    MAXIMUM_EXPIRY_PASSPORT: float = 10.
    """Maximum time required for `passport_expiry_date_remaining`
    """

    MINIMUM_EDUCATION_PERIOD: float = 1.0
    """Minimum time required for `passport_expiry_date_remaining`
    """
    MAXIMUM_EDUCATION_PERIOD: float = 10.
    """Maximum time required for `passport_expiry_date_remaining`
    """

    MAXIMUM_SIBLING_FOREIGNER_COUNT: int = 7
    """Maximum sibling foreigner count
    """

    MAXIMUM_CHILD_COUNT: int = 4
    """Maximum number of children
    """

    MAXIMUM_SIBLING_COUNT: int = 7
    """Maximum number of siblings
    """

    MAXIMUM_PARENT_COUNT: int = 2
    """Obviously!
    """

    MAXIMUM_SPOUSE_COUNT: int = 1
    """Canada isn't a muslim country!
    """

# data used for aggregating SHAP values into categories based on features
class FeatureCategories(CustomNamingEnum):
    """Categories of features based on their meaning

    These values have been provided by the domain expert, and one
    should create their own. Note that, the features for each group
    also has been provided which can be found in 
    :dict:`FEATURE_CATEGORY_TO_FEATURE_NAME_MAP` dictionary.
    """

    PURPOSE = auto()
    EMOTIONAL = auto()
    CAREER = auto()
    FINANCIAL = auto()


FEATURE_CATEGORY_TO_FEATURE_NAME_MAP = {
    FeatureCategories.PURPOSE: [
        'P1.PD.CurrCOR.Row2.Country', 'P1.PD.CurrCOR.Row2.Status', 
        'P1.PD.PrevCOR.Row2.Country', 'P1.PD.PrevCOR.Row3.Country',
        'P1.PD.SameAsCORIndicator', 'P1.PD.CWA.Row2.Country', 
        'P1.PD.CWA.Row2.Status', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit',
        'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe', 
        'P3.cntcts_Row2.Relationship.RelationshipToMe', 'P3.noAuthStay',
        'P3.refuseDeport', 'P3.BGI2.PrevApply', 'P3.DOV.PrpsRow1.HLS.Period',
        'P2.MS.SecA.Psprt.ExpiryDate.Remaining',

        # doubted
        'P1.PD.PrevCOR.Row2.Period', 'P1.PD.PrevCOR.Row3.Period',
        'P1.PD.PrevCOR.Row.Count',
    ],
    FeatureCategories.EMOTIONAL: [
        'P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator', 'P1.PD.Sex.Sex',
        'P2.MS.SecA.PrevMarrIndicator', 'P1.MS.SecA.DateOfMarr.Period',
        'P2.MS.SecA.Period', 'p1.SecA.App.ChdMStatus', 'p1.SecA.Mo.ChdMStatus', 
        'p1.SecA.Fa.ChdMStatus', 'p1.SecB.Chd.[0].ChdMStatus', 'p1.SecB.Chd.[0].ChdRel',
        'p1.SecB.Chd.[1].ChdMStatus', 'p1.SecB.Chd.[1].ChdRel', 'p1.SecB.Chd.[2].ChdMStatus',
        'p1.SecB.Chd.[2].ChdRel', 'p1.SecB.Chd.[3].ChdMStatus', 'p1.SecB.Chd.[3].ChdRel',
        'p1.SecC.Chd.[0].ChdMStatus', 'p1.SecC.Chd.[0].ChdRel', 'p1.SecC.Chd.[1].ChdMStatus',
        'p1.SecC.Chd.[1].ChdRel', 'p1.SecC.Chd.[2].ChdMStatus', 'p1.SecC.Chd.[2].ChdRel',
        'p1.SecC.Chd.[3].ChdMStatus', 'p1.SecC.Chd.[3].ChdRel', 'p1.SecC.Chd.[4].ChdMStatus',
        'p1.SecC.Chd.[4].ChdRel', 'p1.SecC.Chd.[5].ChdMStatus', 'p1.SecC.Chd.[5].ChdRel',
        'p1.SecC.Chd.[6].ChdMStatus', 'p1.SecC.Chd.[6].ChdRel', 'p1.SecA.Sps.SpsDOB.Period',
        'p1.SecA.Mo.MoDOB.Period', 'p1.SecA.Fa.FaDOB.Period', 
        'p1.SecB.Chd.[0].ChdDOB.Period', 'p1.SecB.Chd.[1].ChdDOB.Period', 
        'p1.SecB.Chd.[2].ChdDOB.Period', 'p1.SecB.Chd.[3].ChdDOB.Period',
        'p1.SecC.Chd.[0].ChdDOB.Period', 'p1.SecC.Chd.[1].ChdDOB.Period',
        'p1.SecC.Chd.[2].ChdDOB.Period', 'p1.SecC.Chd.[3].ChdDOB.Period',
        'p1.SecC.Chd.[4].ChdDOB.Period', 'p1.SecC.Chd.[5].ChdDOB.Period',
        'p1.SecC.Chd.[6].ChdDOB.Period',
        'p1.SecB.Chd.X.ChdAccomp.Count', 'p1.SecA.ParAccomp.Count', 
        'p1.SecA.Sps.SpsAccomp.Count', 'p1.SecC.Chd.X.ChdAccomp.Count',
        'p1.SecB.Chd.X.ChdRel.ChdCount', 'p1.SecC.Chd.X.ChdRel.ChdCount',
        'p1.SecX.LongDistAddr',

        # doubted
        'P1.PD.DOBYear.Period', 'p1.SecC.Chd.X.ChdCOB.ForeignerCount',
        'p1.SecB.ChdMoFaSps.X.ChdCOB.ForeignerCount', 'p1.SecX.ForeignAddr',

    ],
    FeatureCategories.CAREER: [
        'P3.Edu.EduIndicator', 'P3.Edu.Edu_Row1.FieldOfStudy', 
        'P3.Occ.OccRow1.Occ.Occ', 'P3.Occ.OccRow2.Occ.Occ', 'P3.Occ.OccRow3.Occ.Occ',
        'P3.Edu.Edu_Row1.Period', 'P3.Occ.OccRow1.Period', 'P3.Occ.OccRow2.Period',
        'P3.Occ.OccRow3.Period', 
        
        # doubted
        'P3.Occ.OccRow1.Country.Country', 'P3.Edu.Edu_Row1.Country.Country',
        'P3.Occ.OccRow2.Country.Country', 'P3.Occ.OccRow3.Country.Country',


    ],
    FeatureCategories.FINANCIAL: [
        'P3.DOV.PrpsRow1.Funds.Funds',

        # doubted
        'P1.PD.CWA.Row2.Period', 
        
    ]
}
"""Dictionary of features belonging to each category

For each group, we have list of names of features that are keys.
Then, we can use these values for instance, for aggregating
these values based on the features for each group.

See Also:
    - :class:`FeatureCategories`
    
Note:
    Some features are not explicitly associated to a single group,
    since they came from EDA process or other heuristics.

    So we have two questions:
        2. how should they be shared?
        1. how to normalize SHAP values after sharing?

    At the moment, each feature that was in question, 
    has been assigned to the *most likely* group, demonstrated
    in parentheses. The list of features in doubt are:

        1. 16  P3.Edu.Edu_Row1.Country.Country: 1 2 (3)
        2. 18  P3.Occ.OccRow1.Country.Country: 1 2 (3)
        3. 20  P3.Occ.OccRow2.Country.Country: 1 2 (3)
        4. 22  P3.Occ.OccRow3.Country.Country: 1 2 (3)
        5. 26  P1.PD.DOBYear.Period: 1 (2)
        6. 27  P1.PD.PrevCOR.Row2.Period: (1) 2
        7. 28  P1.PD.PrevCOR.Row3.Period: (1) 2
        8. 29  P1.PD.CWA.Row2.Period: 1 (4)
        9. 78  P1.PD.PrevCOR.Row.Count: (1) 2
        10. 79  p1.SecC.Chd.X.ChdCOB.ForeignerCount: 1 (2)
        11. 80  p1.SecB.ChdMoFaSps.X.ChdCOB.ForeignerCount: 1 (2)
        12. 88  p1.SecX.ForeignAddr: 1 (2)
    
    First value represents the feature index in database, second value
    represents the name of the feature, and anything after ``:`` demos
    the possible groups that this feature could belong too (the currently 
    assigned group is indicated via parenthesis e.g. ``(x)``).
 """


FEATURE_NAME_TO_TEXT_MAP = {
    'P3.DOV.PrpsRow1.Funds.Funds': 'شرایط تمکن مالی',
    'P1.PD.DOBYear.Period': 'شرایط سنی',
    'P1.PD.CWA.Row2.Period': 'مدت زمان اقامت شما در کشور انگشت نگاری',
    'P1.MS.SecA.DateOfMarr.Period': 'مدت زمان نعهل فعلی',
    'P2.MS.SecA.Period': 'مدت زمان مزدوج بودن در ازدواج قبلی',
    'P2.MS.SecA.Psprt.ExpiryDate.Remaining': 'مدت زمان باقی مانده از اعتبار پاسپورت',
    'P3.DOV.PrpsRow1.HLS.Period': 'مدت زمان اقامت',
    'P3.Edu.Edu_Row1.Period': 'مدت تحصیلات و آموزش حرفه ای',
    'P3.Occ.OccRow1.Period': 'مدت اشتغال به آخرین شغل',
    'P3.Occ.OccRow2.Period': 'مدت اشتغال به شغل قبلی',
    'P3.Occ.OccRow3.Period': 'مدت اشتغال به شغل قبلی',
    'p1.SecA.App.ChdMStatus': 'وضعیت تعهل',
    'p1.SecB.Chd.X.ChdDOB.Period.Average': 'متوسط سن فرزندان',
    'p1.SecC.Chd.X.ChdDOB.Period.Average': 'متوسط سن برادران و خواهران',
    'P1.PD.Sex.Sex_Female': 'جنسیت زن',
    'P1.PD.Sex.Sex_Male': 'جنسیت مرد',
    'P2.MS.SecA.PrevMarrIndicator_False': 'عدم سابقه ازدواج قبلی',
    'P2.MS.SecA.PrevMarrIndicator_True': 'سابقه ازدواج قبلی',
    'P3.noAuthStay_False': 'عدم سابقه حضور بیش از مدت قانونی',
    'P3.noAuthStay_True': 'سابقه حضور بیش از مدت قانونی',
    'P3.refuseDeport_False': 'عدم سابقه رد یا دیپورتی ویزا',
    'P3.refuseDeport_True': 'سابقه رد یا دیپورتی ویزا',
    'P3.BGI2.PrevApply_False': 'نداشتن سابقه اقدام قبلی برای ویزا',
    'P3.BGI2.PrevApply_True': 'انجام اقدام قبلی برای ویزا',
    'x2_6': 'وضعیت تعهل مزدوج',
    'x2_3': 'وضعیت تعهل مجرد',
    'P1.PD.CWA.Row2.Status_1': 'وضعیت توریستی در کشور انگشت نگاری',
    'P1.PD.CWA.Row2.Country_TURKEY': 'ترکیه به عنوان  کشور انگشت نگاری',
    'P1.PD.CWA.Row2.Country_OTHER': 'سایر کشورها به عنوان  کشور انگشت نگاری',
    'P1.PD.CWA.Row2.Country_Georgia': 'گرجستان به عنوان  کشور انگشت نگاری',
    'P1.PD.CWA.Row2.Country_UAE': 'امارات به عنوان  کشور انگشت نگاری',
    'P1.PD.CWA.Row2.Country_Armenia': 'ارمنستان به عنوان  کشور انگشت نگاری',
    'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit_family visit':  'هدف شما از سفر برای ملاقات خانواده',
    'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit_tourism': 'هدف شما از سفر توریسم',
    'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit_visit': 'هدف شما از ملاقات دوستان و اوقام دور',
    'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit_other': 'هدف سایر شما از سفر',
    'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit_business': 'هدف تجاری شما از سفر',
    'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe_f1': 'ملاقات اقوام درجه اول به عنوان ارتباط شما در هدف از سفر',
    'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe_f2': 'ملاقات اقوام درجه دوم به عنوان ارتباط شما در هدف از سفر',
    'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe_friend': 'ملاقات دوستانتان به عنوان ارتباط شما در هدف سفر',
    'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe_hotel': 'رفتن به هتل به عنوان ارتباط شما در هدف از سفر',
    'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe_ukn': 'مقصد نامتعارف به عنوان ارتباط شما در هدف از سفر',
    'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe_work': 'مقصد کاری شما در هدف از سفر',
    'P3.cntcts_Row2.Relationship.RelationshipToMe_f1': 'ملاقات اقوام درجه اول به عنوان ارتباط شما در هدف  دوم از سفر',
    'P3.cntcts_Row2.Relationship.RelationshipToMe_f2': 'ملاقات اقوام درجه دوم به عنوان ارتباط شما در هدف دوم از سفر',
    'P3.cntcts_Row2.Relationship.RelationshipToMe_friend': 'ملاقات دوستانتان به عنوان ارتباط شما در هدف دوم از سفر',
    'P3.cntcts_Row2.Relationship.RelationshipToMe_hotel': 'رفتن به هتل به عنوان ارتباط شما در هدف دوم از سفر',
    'P3.cntcts_Row2.Relationship.RelationshipToMe_ukn': 'مقصد نامتعارف به عنوان ارتباط شما در هدف دوم از سفر',
    'P2.MS.SecA.Psprt.ExpiryDate.Remaining': 'مدت اعتبار پاسپورت',
    'P3.Edu.Edu_Row1.FieldOfStudy_unedu': 'عدم داشتن تحصیلات',
    'P3.Edu.Edu_Row1.FieldOfStudy_master': 'داشتن تحصیلات کارشناسی ارشد',
    'P3.Edu.Edu_Row1.FieldOfStudy_bachelor': 'داشتن تحصیلات کارشناسی',
    'P3.Edu.Edu_Row1.FieldOfStudy_phd': 'داشتن تحصیلات دکترا',
    'P3.Edu.Edu_Row1.FieldOfStudy_apprentice': 'داشتن تحصیلات کاردانی، فنی و حرفه ای',
    'P3.Edu.Edu_Row1.FieldOfStudy_diploma': 'داشتن دیپلم های تحصیلی و فنی',
    'P3.Occ.OccRow1.Occ.Occ_employee': 'وضعیت شغلی به عنوان کارمند',
    'P3.Occ.OccRow1.Occ.Occ_OTHER': 'وضعیت شغلی نا معلوم',
    'P3.Occ.OccRow1.Occ.Occ_student': 'عدم اشتغال و دانش آموز بودن',
    'P3.Occ.OccRow1.Occ.Occ_housewife': 'عدم اشتفال و خانه دار بودن',
    'P3.Occ.OccRow1.Occ.Occ_manager': 'وضعیت شغلی به عنوان مدیر',
    'P3.Occ.OccRow1.Occ.Occ_retired': 'وضعیت شغلی به عنوان بازنشسته',
    'P3.Occ.OccRow1.Occ.Occ_specialist': 'وضعیت شغلی به عنوان متخصص حرفه ای',
    'P3.Occ.OccRow2.Occ.Occ_employee': 'وضعیت شغلی اسبق به عنوان کارمند',
    'P3.Occ.OccRow2.Occ.Occ_OTHER': 'وضعیت شغلی اسبق نا معلوم',
    'P3.Occ.OccRow2.Occ.Occ_student': 'عدم اشتغال و دانش آموز بودن شما در گذشته',
    'P3.Occ.OccRow2.Occ.Occ_housewife': 'عدم اشتفال و خانه دار بودن در گذشته',
    'P3.Occ.OccRow2.Occ.Occ_manager': 'وضعیت اسبق شغلی به عنوان مدیر',
    'P3.Occ.OccRow2.Occ.Occ_retired': 'وضعیت اسبق شغلی به عنوان بازنشسته',
    'P3.Occ.OccRow2.Occ.Occ_specialist': 'وضعیت اسبق شغلی به عنوان متخصص حرفه ای',
    'P3.Occ.OccRow3.Occ.Occ_employee': 'وضعیت شغلی سابق به عنوان کارمند',
    'P3.Occ.OccRow3.Occ.Occ_OTHER': 'وضعیت شغلی سابق نا معلوم',
    'P3.Occ.OccRow3.Occ.Occ_student': 'عدم اشتغال و دانش آموز بودن شما در گذشته',
    'P3.Occ.OccRow3.Occ.Occ_housewife': 'عدم اشتفال و خانه دار بودن در گذشته',
    'P3.Occ.OccRow3.Occ.Occ_manager': 'وضعیت سابق شغلی به عنوان مدیر',
    'P3.Occ.OccRow3.Occ.Occ_retired': 'وضعیت سابق شغلی به عنوان بازنشسته',
    'P3.Occ.OccRow3.Occ.Occ_specialist': 'وضعیت سابق شغلی به عنوان متخصص حرفه ای',
    'P1.PD.PrevCOR.Row.Count': 'تعداد کشورهایی که در گذشته در آن اقامت داشته اید',
    'p1.SecC.Chd.X.ChdCOB.ForeignerCount': 'تعداد خواهران/برادرانی کهدر خارج از کشور دارید',
    'p1.SecB.ChdMoFaSps.X.ChdCOB.ForeignerCount': 'تعداد فرزندان، پدر و مادر که در خارج از کشور دارید',
    'p1.SecB.Chd.X.ChdAccomp.Count': 'تعداد فرزندان شما که همراه شما هستند',
    'p1.SecA.ParAccomp.Count': 'همراهی پدر و مادر شما',
    'p1.SecA.Sps.SpsAccomp.Count': 'همراهی همسر شما',
    'p1.SecC.Chd.X.ChdAccomp.Count': 'تعداد برادران و خواهرانی که همراه شما هستند',
    'p1.SecB.Chd.X.ChdRel.ChdCount': 'تعداد فرزندان شما',
    'p1.SecC.Chd.X.ChdRel.ChdCount': 'تعداد برادران و خواهران شما',
    'p1.SecX.LongDistAddr': 'تعداد اعضای خانواده ای که در خارج از شهر شما سکونت دارند',
    'p1.SecX.ForeignAddr': 'تعداد اعضای خانواده ای که در خارج از کشور سکونت دارند',
}
"""Dictionary of features names to their textual descriptions

This dictionary is used for generating descriptions
with respect to the XAI values so we can tell the users
why such decision is made textual manner. 

Note:
    Given that generating "beautiful" textual description is a
    open problem in NLP domain (LLMs are doing a great job),
    we have decided to use this dictionary for now instead of relying
    on such large models for this simple task. For now, we devise
    keywords that describe each feature, and use XAI value for generate
    adjective (e.g., negative XAI value -> bad, positive XAI value -> good)
    that are concatenated to textual keywords here.
"""
