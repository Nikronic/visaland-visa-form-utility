__all__ = [
    'PredictionResponse', 'Payload',
    'CountryNamesResponse', 'CanadaMarriageStatusResponse', 'SiblingRelationResponse',
    'ChildRelationResponse', 'CanadaContactRelationResponse', 'CanadaResidencyStatusResponse',
    'EducationFieldOfStudyResponse'
]

# core
import pydantic
from pydantic import validator
import json
# ours
from vizard.data.constant import (
    CanadaMarriageStatus,
    SiblingRelation,
    ChildRelation,
    CanadaContactRelation,
    CanadaResidencyStatus,
    Sex,
    EducationFieldOfStudy
)
# helpers
from typing import Any, List


class BaseModel(pydantic.BaseModel):
    """
    Reference:
        * https://stackoverflow.com/a/70640522/18971263
    """
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

    @classmethod
    def __get_validators__(cls):
        yield cls._validate_from_json_string

    @classmethod
    def _validate_from_json_string(cls, value):
        if isinstance(value, str):
            return cls.validate(json.loads(value.encode()))
        return cls.validate(value)


class PredictionResponse(BaseModel):
    result: float


class Payload(BaseModel):
    alias_name_indicator: bool = False

    sex: str
    @validator('sex')
    def _sex(cls, value):
        if value not in Sex.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value

    current_country_of_residence_country: str = 'iran'
    current_country_of_residence_status: str = 'citizen'
    previous_country_of_residence_country2: str = 'iran'
    previous_country_of_residence_country3: str = 'iran'

    same_as_country_of_residence_indicator: bool = False
    country_where_applying_country: str = 'TURKEY'
    country_where_applying_status: str = 'OTHER'

    @validator(
        'current_country_of_residence_status',
        'country_where_applying_status'
    )
    def _residence_status(cls, value):
        value = value.lower().strip()
        if value not in CanadaResidencyStatus.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value

    previous_marriage_indicator: bool = False

    purpose_of_visit: str = 'tourism'
    funds: float = 8000.
    @validator('funds')
    def _funds(cls, value):
        if value <= 0.:
            raise ValueError('funds cannot be negative number.')
        return value

    contact_relation_to_me: str = 'hotel'
    contact_relation_to_me2: str = 'ukn'
    @validator(
        'contact_relation_to_me',
        'contact_relation_to_me2'
    )
    def _contact_relation_to_me(cls, value):
        value = value.lower().strip()
        if value not in CanadaContactRelation.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value

    education_indicator: bool = False

    education_field_of_study: str = 'unedu'
    @validator('education_field_of_study')
    def _education_field_of_study(cls, value):
        value = value.lower().strip()
        if value not in EducationFieldOfStudy.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value
    
    education_country: str = 'Unknown'
    
    occupation_title1: str = 'OTHER'
    occupation_country1: str = 'iran'
    occupation_title2: str = 'OTHER'
    occupation_country2: str = 'iran'
    occupation_title3: str = 'OTHER'
    occupation_country3: str = 'iran'
    
    no_authorized_stay: bool = False
    refused_entry_or_deport: bool = False
    previous_apply: bool = False

    date_of_birth: float
    @validator('date_of_birth')
    def _date_of_birth(cls, value):
        if value < 18:
            raise ValueError('This service only accepts adults')
        return value

    previous_country_of_residency_period2: float = 0  # years
    previous_country_of_residency_period3: float = 0  # years
    @validator(
        'previous_country_of_residency_period2',
        'previous_country_of_residency_period3'
    )
    def _previous_country_of_residency_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    country_where_applying_period: float = 30.  # days
    @validator('country_where_applying_period')
    def _country_where_applying_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    marriage_period: float = 0.  # years
    previous_marriage_period: float = 0.  # years
    @validator(
        'marriage_period',
        'previous_marriage_period'
    )
    def _marriage_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    passport_expiry_date_remaining: float = 3.  # years
    @validator('passport_expiry_date_remaining')
    def _passport_expiry_date_remaining(cls, value):
        if (value < 0) and (value > 10):
            raise ValueError('Value cannot be negative or > 10')
        return value
    
    how_long_stay_period: float = 30.  # days
    @validator('how_long_stay_period')
    def _how_long_stay_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    education_period: float = 0.  # years
    @validator('education_period')
    def _education_period(cls, value):
        if (value < 0) and (value > 10):
            raise ValueError('Value cannot be negative')
        return value

    occupation_period: float
    occupation_period2: float = 0.  # years    
    occupation_period3: float = 0.  # years
    @validator(
        'occupation_period',
        'occupation_period2',
        'occupation_period3'
    )
    def _occupation_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    applicant_marital_status: str = 'single'
    mother_marital_status: str = 'married'
    father_marital_status: str = 'married'

    child_marital_status0: str = 'unknown'
    child_relation0: str = 'other'
    child_marital_status1: str = 'unknown'
    child_relation1: str = 'other'
    child_marital_status2: str = 'unknown'
    child_relation2: str = 'other'
    child_marital_status3: str = 'unknown'
    child_relation3: str = 'other'

    @validator(
        'child_relation0',
        'child_relation1',
        'child_relation2',
        'child_relation3',
    )
    def _child_relation(cls, value):
        value = value.lower().strip()
        if value not in ChildRelation.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value
    
    sibling_marital_status0: str = 'unknown'
    sibling_relation0: str = 'other'
    sibling_marital_status1: str = 'unknown'
    sibling_relation1: str = 'other'
    sibling_marital_status2: str = 'unknown'
    sibling_relation2: str = 'other'
    sibling_marital_status3: str = 'unknown'
    sibling_relation3: str = 'other'
    sibling_marital_status4: str = 'unknown'
    sibling_relation4: str = 'other'
    sibling_marital_status5: str = 'unknown'
    sibling_relation5: str = 'other'
    sibling_marital_status6: str = 'unknown'
    sibling_relation6: str = 'other'

    @validator(
        'sibling_relation0',
        'sibling_relation1',
        'sibling_relation2',
        'sibling_relation3',
        'sibling_relation4',
        'sibling_relation5',
        'sibling_relation6',
    )
    def _sibling_relation(cls, value):
        value = value.lower().strip()
        if value not in SiblingRelation.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value

    @validator(
        'applicant_marital_status',
        'mother_marital_status',
        'father_marital_status',
        'child_marital_status0',
        'child_marital_status1',
        'child_marital_status2',
        'child_marital_status3',
        'sibling_marital_status0',
        'sibling_marital_status1',
        'sibling_marital_status2',
        'sibling_marital_status3',
        'sibling_marital_status4',
        'sibling_marital_status5',
        'sibling_marital_status6',
    )
    def _marital_status(cls, value):
        value = value.lower().strip()
        if value not in CanadaMarriageStatus.get_member_names():
            raise ValueError(f'"{value}" is not valid')
        return value

    spouse_date_of_birth: float = 0.  # years
    mother_date_of_birth: float
    father_date_of_birth: float
    
    child_date_of_birth0: float = 0.  # years
    child_date_of_birth1: float = 0.  # years
    child_date_of_birth2: float = 0.  # years
    child_date_of_birth3: float = 0.  # years

    sibling_date_of_birth0: float = 0.  # years
    sibling_date_of_birth1: float = 0.  # years
    sibling_date_of_birth2: float = 0.  # years
    sibling_date_of_birth3: float = 0.  # years
    sibling_date_of_birth4: float = 0.  # years
    sibling_date_of_birth5: float = 0.  # years
    sibling_date_of_birth6: float = 0.  # years

    @validator(
        'spouse_date_of_birth',
        'mother_date_of_birth',
        'father_date_of_birth',

        'child_date_of_birth0',
        'child_date_of_birth1',
        'child_date_of_birth2',
        'child_date_of_birth3',

        'sibling_date_of_birth0',
        'sibling_date_of_birth1',
        'sibling_date_of_birth2',
        'sibling_date_of_birth3',
        'sibling_date_of_birth4',
        'sibling_date_of_birth5',
        'sibling_date_of_birth6',
    )
    def _kin_date_of_birth(cls, value):
        if value < 0:
            raise ValueError(f'Value cannot be negative')
        return value

    previous_country_of_residence_count: int = 0
    @validator('previous_country_of_residence_count')
    def _previous_country_of_residence_count(cls, value):
        if (value < 0) and (value > 5):
            raise ValueError('Value cannot be negative or > 5')
        return value

    sibling_foreigner_count: int = 0
    @validator('sibling_foreigner_count')
    def _sibling_foreigner_count(cls, value):
        if (value < 0) and (value > 7):
            raise ValueError('Value cannot be negative or > 7')
        return value

    child_mother_father_spouse_foreigner_count: int = 0
    @validator('child_mother_father_spouse_foreigner_count')
    def _child_mother_father_spouse_foreigner_count(cls, value):
        if (value < 0) and (value > 4 + 2 + 1):
            raise ValueError('Value cannot be negative or > 4 + 2 + 1')
        return value

    child_accompany: int = 0
    @validator('child_accompany')
    def _child_accompany(cls, value):
        if (value < 0) and (value > 4):
            raise ValueError('Value cannot be negative or > 4')
        return value

    parent_accompany: int = 0
    @validator('parent_accompany')
    def _parent_accompany(cls, value):
        if (value < 0) and (value > 2):
            raise ValueError('Value cannot be negative or > 2')
        return value

    spouse_accompany: int = 0
    @validator('spouse_accompany')
    def _spouse_accompany(cls, value):
        if (value < 0) and (value > 1):
            raise ValueError(
                'Value cannot be negative no matter how much u hate your spouse'
                ' Or bigger than one (having multiple spouse is a bad thing!)'
            )
        return value
    sibling_accompany: int = 0
    @validator('sibling_accompany')
    def _sibling_accompany(cls, value):
        if (value < 0) and (value > 7):
            raise ValueError('Value cannot be negative or > 7')
        return value

    child_count: int = 0
    @validator('child_count')
    def _child_count(cls, value):
        if (value < 0) and (value > 4):
            raise ValueError('Value cannot be negative or > 4')
        return value
    
    sibling_count: int = 0
    @validator('sibling_count')
    def _sibling_count(cls, value):
        if (value < 0) and (value > 7):
            raise ValueError('Value cannot be negative or > 7')
        return value

    long_distance_child_sibling_count: int = 0
    @validator('long_distance_child_sibling_count')
    def _long_distance_child_sibling_count(cls, value):
        if (value < 0) and (value > 7 + 4):
            raise ValueError('Value cannot be negative or > 7 + 4')
        return value
    
    foreign_living_child_sibling_count: int = 0
    @validator('foreign_living_child_sibling_count')
    def _foreign_living_child_sibling_count(cls, value):
        if (value < 0) and (value > 7 + 4):
            raise ValueError('Value cannot be negative or > 7 + 4')
        return value

    class Config:
        orm_mode = True


class CountryNamesResponse(BaseModel):
    """Country names used in our application

    Note:
        Country names are extracted from :class:`vizard.data.preprocessor.WorldBankXMLProcessor`
        which are country names used in WorldBank dataset.

    """

    country_names: List[str]


class CanadaMarriageStatusResponse(BaseModel):
    """Canada marriage status states in string format

    Note:
        See :class:`vizard.data.constant.CanadaMarriageStatus` for more info
        for possible values.
    """

    marriage_status_types: List[str]

class SiblingRelationResponse(BaseModel):
    """Sibling relation types names

    Note:
        See :class:`vizard.data.constant.SiblingRelation` for more info
        for possible values.
    """

    sibling_relation_types: List[str]

class ChildRelationResponse(BaseModel):
    """Child relation types names

    Note:
        See :class:`vizard.data.constant.ChildRelation` for more info
        for possible values.
    """

    child_relation_types: List[str]

class CanadaContactRelationResponse(BaseModel):
    """Contact relation types names in Canada

    Note:
        See :class:`vizard.data.constant.CanadaContactRelation` for more info
        for possible values.
    """

    canada_contact_relation_types: List[str]

class CanadaResidencyStatusResponse(BaseModel):
    """Residency status types names in Canada

    Note:
        See :class:`vizard.data.constant.CanadaResidencyStatus` for more info
        for possible values.
    """

    canada_residency_status_types: List[str]


class EducationFieldOfStudyResponse(BaseModel):
    """Education field of study types names

    Note:
        See :class:`vizard.data.constant.EducationFieldOfStudy` for more info
        for possible values.
    """

    education_field_of_study_types: List[str]
