# core
import pydantic
import json
# helpers
from typing import List, Any


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
    # TODO: make sure default values are in correct type and value
    alias_name_indicator: bool = False
    sex: str

    current_country_of_residence_country: str = 'iran'
    current_country_of_residence_status: str = 'citizen'
    previous_country_of_residence_country2: str = 'iran'
    previous_country_of_residence_country3: str = 'iran'

    same_as_country_of_residence_indicator: bool = False
    country_where_applying_country: str = 'TURKEY'
    country_where_applying_status: str = 'OTHER'

    previous_marriage_indicator: bool = False

    purpose_of_visit: str = 'tourism'
    funds: float = 8000.
    contact_relation_to_me: str = 'hotel'
    contact_relation_to_me2: str = 'hotel'

    education_indicator: bool = False
    education_field_of_study: str = 'OTHER'
    education_country: str = 'iran'
    
    occupation_title1: str = 'employee'
    occupation_country1: str = 'iran'
    occupation_title2: str = 'OTHER'
    occupation_country2: str = 'iran'
    occupation_title3: str = 'OTHER'
    occupation_country3: str = 'iran'
    
    no_authorized_stay: bool = False
    refused_entry_or_deport: bool = False
    previous_apply: bool = False

    date_of_birth: float

    previous_country_of_residency_period2: float = 0
    previous_country_of_residency_period3: float = 0

    country_where_applying_period: float = 30.  # days

    marriage_period: float = 0.
    previous_marriage_period: float = 0.

    passport_expiry_date_remaining: float = 3.  # years?
    how_long_stay_period: float = 30.  # days

    education_period: float = 0.

    occupation_period: float
    occupation_period2: float = 0.
    occupation_period3: float = 0.

    applicant_marital_status: str = 'single'
    mother_marital_status: str = 'married'
    father_marital_status: str = 'married'

    child_marital_status0: str = 'OTHER'
    child_relation0: str = 'OTHER'
    child_marital_status1: str = 'OTHER'
    child_relation1: str = 'OTHER'
    child_marital_status2: str = 'OTHER'
    child_relation2: str = 'OTHER'
    child_marital_status3: str = 'OTHER'
    child_relation3: str = 'OTHER'

    sibling_marital_status0: str = 'OTHER'
    sibling_relation0: str = 'OTHER'
    sibling_marital_status1: str = 'OTHER'
    sibling_relation1: str = 'OTHER'
    sibling_marital_status2: str = 'OTHER'
    sibling_relation2: str = 'OTHER'
    sibling_marital_status3: str = 'OTHER'
    sibling_relation3: str = 'OTHER'
    sibling_marital_status4: str = 'OTHER'
    sibling_relation4: str = 'OTHER'
    sibling_marital_status5: str = 'OTHER'
    sibling_relation5: str = 'OTHER'
    sibling_marital_status6: str = 'OTHER'
    sibling_relation6: str = 'OTHER'

    spouse_date_of_birth: float = 0.
    mother_date_of_birth: float
    father_date_of_birth: float 
    
    child_date_of_birth0: float = 0
    child_date_of_birth1: float = 0
    child_date_of_birth2: float = 0
    child_date_of_birth3: float = 0

    sibling_date_of_birth0: float = 0
    sibling_date_of_birth1: float = 0
    sibling_date_of_birth2: float = 0
    sibling_date_of_birth3: float = 0
    sibling_date_of_birth4: float = 0
    sibling_date_of_birth5: float = 0
    sibling_date_of_birth6: float = 0

    previous_country_of_residence_count: int = 0

    sibling_foreigner_count: int = 0
    child_mother_father_spouse_foreigner_count: int = 0

    child_accompany: int = 0
    parent_accompany: int = 0
    spouse_accompany: bool = False
    sibling_accompany: int = 0

    child_count: int = 0
    sibling_count: int = 0

    long_distance_child_sibling_count: int = 0
    foreign_living_child_sibling_count: int = 0

    class Config:
        orm_mode = True
