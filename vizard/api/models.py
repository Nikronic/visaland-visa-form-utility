__all__ = [
    'PredictionResponse', 'Payload', 'ConstantStatesResponse', 
    'XaiResponse', 'XaiAggregatedGroupResponse', 'PotentialResponse'
]

# core
import pydantic
from pydantic import field_validator
from pydantic.fields import FieldInfo
import json
# ours
from vizard.data.constant import (
    CanadaMarriageStatus,
    CanadaContactRelation,
    CanadaResidencyStatus,
    Sex,
    EducationFieldOfStudy,
    CountryWhereApplying,
    PurposeOfVisit,
    OccupationTitle,
    CanadaGeneralConstants,
    FeatureCategories
)
# helpers
from typing import Any, Dict, List, Tuple, Type, Optional


class BaseModel(pydantic.BaseModel):
    """Extension of :class:`pydantic.BaseModel` to parse ``File`` and ``Form`` data along side each other

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
            return cls.model_validate(json.loads(value.encode()))
        return cls.model_validate(value)


class PredictionResponse(BaseModel):
    """Response model for the prediction of machine learning model

    Explanation of variables:

        * ``result``: is the final prediction probability by the ML model
        * ``next_variable``: is the name of the variable with highest effect on
            the ``result``. This is the variable with highest XAI value 
            as part of the request body.

    Note:
        This is the final goal of the project from technical aspect
    """
    result: float
    next_variable: str

    
def validate_model_fields(
        model: BaseModel,
        fields2docs: Dict[str, str]
    ) -> None:
    """Checks if all fields of the pydantic model is covered by ``fields2docs``

    In other words, ``model`` 's  fields must always be a subset of ``fields2docs``
    since ``fields2docs`` contains the documentation for all the fields (i.e., the
    entire features/columns/attributes)

    Args:
        model (BaseModel): Pydantic model representing the input or output of the API
        fields2docs (Dict[str, str]): A dictionary of ``model`` 's fields to their 
            documentation string.

    Raises:
        ValueError: If there is a field in ``model`` that is not available in ``fields2docs``
    """

    # get the model fields
    model_fields: Dict[str, FieldInfo] = model.model_fields
    # check subset
    if not model_fields.keys() <= fields2docs.keys():
        raise ValueError('Not all model fields are covered by ``fields2docs``.')


payload_fields2docs: Dict[str, str] = {
    'sex': 'The sex of the user. Input must be one of ["Female", "Male"]', 
    'country_where_applying_country': 'The country the applicants goes for fingerprint. Input must be a string in UPPERCASE.', 
    'country_where_applying_status': 'The residency status of the applicant in the country the they go for fingerprint. Input must be a string in ["citizen", "visitor", "OTHER"]', 
    'previous_marriage_indicator': 'Whether or not the user has previous marriage. `true` or `false`.', 
    'purpose_of_visit': 'The purpose of visit. Input must be one of ["family visit", "visit", "tourism", "other"]', 
    'funds': 'How much money the user is bringing. Input must be a single float.', 
    'contact_relation_to_me': 'When user is visiting, who the contact is. Input must be one of ["f1", "f2", "friend", "hotel", "ukn"]', 
    'contact_relation_to_me2': 'When user is visiting, who the contact is. Input must be one of ["f1", "f2", "friend", "hotel", "ukn"]', 
    'education_field_of_study': 'Study level of the user. Input must be one of ["phd", "master", "bachelor", "diploma", "apprentice", "unedu"]', 
    'occupation_title1': 'Occupation of the user. Input must be one of ["manager", "specialist", "employee", "retired", "student", "OTHER"]', 
    'occupation_title2': 'Occupation of the user. Input must be one of ["manager", "specialist", "employee", "retired", "student", "OTHER"]', 
    'occupation_title3': 'Occupation of the user. Input must be one of ["manager", "specialist", "employee", "retired", "student", "OTHER"]', 
    'no_authorized_stay': 'Has the user had stayed longer than they should? Input must be `true` or `false`.', 
    'refused_entry_or_deport': 'Has the user denied entry or deported? Input must be `true` or `false`', 
    'previous_apply': 'Has the user previously applied for the visa? Input must be `true` or `false`', 
    'date_of_birth': 'The age of the person. Input must be a single float in years larger than 18.', 
    'country_where_applying_period': 'How long the user stays in the country the applicants goes for fingerprint. Must be a single integer in days.', 
    'applicant_marital_status': 'The marital status of the user. Input must be one of ["single", "married", "divorced", "widowed", "unknown"]', 
    'marriage_period': 'How long the user has been married. Input must be a single float in years (zero if not married).', 
    'previous_marriage_period': 'How long the user has been married previously (no longer in the same marriage). Input must be a single float in years.', 
    'passport_expiry_date_remaining': 'How much longer the passport is valid. Input must be a single float in years between 1 and 5.', 
    'how_long_stay_period': 'How long the user will be staying. Input must be a single integer in days.', 
    'education_period': 'The study length of the user in the last degree. Input must be a single float in years.', 
    'occupation_period': 'How long the user has been doing the `occupation_title1` job. Input must be a single float in years.', 
    'occupation_period2': 'How long the user has been doing the `occupation_title2` job. Input must be a single float in years.', 
    'occupation_period3': 'How long the user has been doing the `occupation_title3` job. Input must be a single float in years.', 
    'previous_country_of_residence_count': 'How many other countries the user have resided previously. Input must be a single integer.', 
    'sibling_foreigner_count': 'How many siblings of the user are residing in a foreign country. Input must be a single integer.', 
    'child_mother_father_spouse_foreigner_count': 'How many of first-degree relatives are residing in a foreign country. Input must be a single integer.', 
    'child_accompany': 'How many of the users children are accompanying the user. Input must be a single integer.', 
    'parent_accompany': 'Are the parents of the user accompanying him/her? Input must be a single integer between 0 to 2. 0 means none, 2 means both father and mother.', 
    'spouse_accompany': 'Is the spouse of the user accompanying him/her? Input must be either 0 or 1 (you can use `true` or `false`).', 
    'sibling_accompany': 'How many of the siblings are accompanying the user. Input must be a integer (0 if none accompanying).', 
    'child_average_age': 'The average age of children of the user is. Input must a float (0 if no children).', 
    'child_count': 'How many children the user has. Input must be a integer (0 if no children).', 
    'sibling_average_age': 'The average age of siblings of the user is. Input must a float (0 if no sibling).', 
    'sibling_count': 'How many siblings the user has. Input must be a integer (0 if no siblings).', 
    'long_distance_child_sibling_count': 'How many of the children or siblings are residing outside of the city the user is currently residing in. Input must be a integer (0 if none).', 
    'foreign_living_child_sibling_count': 'How many of the children or siblings are residing in a foreign country. Input must be a integer (0 if none).',
}
"""A dictionary of input payload and their documentation for OpenAPI docs

The keys are the fields of :class:`Payload` and values are the description (documentation)
for each field. This dictionary is used to extend the `json_schema` generated by ``Pydantic``
to include these documentations.

See Also:
    class `Config` inside :class:`Payload`.
"""


class Payload(BaseModel):
    # required for having dynamic request body for populating ``next_variable`` of `ResponseModel`
    __slots__ = ('provided_variables',)

    sex: str
    @field_validator('sex')
    def _sex(cls, value):
        if value not in Sex.get_member_names():
            raise ValueError(
                f'"{value}" is not valid.'
                f' Only "{Sex.get_member_names()}" are available.')
        return value

    country_where_applying_country: str = 'TURKEY'
    @field_validator('country_where_applying_country')
    def _country_where_applying_country(cls, value):
        if value.upper() not in CountryWhereApplying.get_member_names():
            raise ValueError(
                f'Country "{value}" is not valid in this system.')
        return value

    country_where_applying_status: str | float | int = 6
    @field_validator('country_where_applying_status')
    def _residence_status(cls, value):
        # convert residency status string to code
        # Residency code `{1: 'citizen', 3: 'visitor', 6: 'other'}`
        if isinstance(value, str):
            value = value.lower().strip()
            if value not in CanadaResidencyStatus.get_member_names():
                raise ValueError(
                    f'"{value}" is not valid'
                    f' Please use one of "{CanadaResidencyStatus.get_member_names()}"')
            return value
        elif isinstance(value, int) or isinstance(value, float):
            value = int(value)
            # get Enum values # TODO: use const class or dict
            member_values: List[int] = []
            for member_ in CanadaResidencyStatus._member_names_:
                member_values.append(CanadaResidencyStatus[member_].value)
            if value not in member_values:
                raise ValueError(
                    f'"{value}" is not valid'
                    f' Please use one of "{member_values}"')
            return value
        return value

    previous_marriage_indicator: bool = False
    @field_validator('previous_marriage_indicator')
    def _previous_marriage_indicator(cls, value):
        transformed_value: bool = False
        if isinstance(value, str):
            transformed_value = True if value.lower() == 'true' else False
        else:
            transformed_value = value
        return transformed_value


    purpose_of_visit: str = 'tourism'
    @field_validator('purpose_of_visit')
    def _purpose_of_visit(cls, value):
        value = value.lower()
        if value.lower() not in PurposeOfVisit.get_member_names():
            raise ValueError(
                f'"{value}" is not valid'
                f' Please use one of "{PurposeOfVisit.get_member_names()}"')
        return value

    funds: float = 8000.
    @field_validator('funds')
    def _funds(cls, value):
        if isinstance(value, str):
            if not value.isnumeric():
                raise ValueError('funds must be a number.')
        if value <= 0.:
            raise ValueError('funds cannot be negative number.')
        return value

    contact_relation_to_me: str = 'hotel'
    contact_relation_to_me2: str = 'ukn'
    @field_validator(
        'contact_relation_to_me',
        'contact_relation_to_me2'
    )
    def _contact_relation_to_me(cls, value):
        value = value.lower().strip()
        if value not in CanadaContactRelation.get_member_names():
            raise ValueError(
                f'"{value}" is not valid'
                f' Please use one of "{CanadaContactRelation.get_member_names()}"')
        return value

    education_field_of_study: str = 'unedu'
    @field_validator('education_field_of_study')
    def _education_field_of_study(cls, value):
        value = value.lower().strip()
        if value not in EducationFieldOfStudy.get_member_names():
            raise ValueError(
                f'"{value}" is not valid'
                f' Please use one of "{EducationFieldOfStudy.get_member_names()}"')
        return value

    occupation_title1: str = 'OTHER'
    occupation_title2: str = 'OTHER'
    occupation_title3: str = 'OTHER'
    @field_validator(
        'occupation_title1',
        'occupation_title2',
        'occupation_title3'
    )
    def _occupation_title(cls, value):
        if value not in OccupationTitle.get_member_names():
            raise ValueError(
                f'"{value}" is not valid'
                f' Please use one of "{OccupationTitle.get_member_names()}"'
            )
        return value

    no_authorized_stay: bool = False
    refused_entry_or_deport: bool = False
    previous_apply: bool = False

    date_of_birth: float = 18.
    @field_validator('date_of_birth')
    def _date_of_birth(cls, value):
        if value < CanadaGeneralConstants.MINIMUM_AGE:
            raise ValueError('This service only accepts adults')
        return value

    country_where_applying_period: float = 30.  # days
    @field_validator('country_where_applying_period')
    def _country_where_applying_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    marriage_period: float = 0.  # years
    previous_marriage_period: float = 0.  # years

    @field_validator(
        'marriage_period',
        'previous_marriage_period'
    )
    def _marriage_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    passport_expiry_date_remaining: float = 3.  # years
    @field_validator('passport_expiry_date_remaining')
    def _passport_expiry_date_remaining(cls, value):
        if (value < CanadaGeneralConstants.MINIMUM_EXPIRY_PASSPORT):
            raise ValueError('The expiry date of your passport is too low.')
        if (value > CanadaGeneralConstants.MAXIMUM_EXPIRY_PASSPORT):
            raise ValueError('A passport cannot legally have such long expiry date.')
        return value

    how_long_stay_period: float = 30.  # days
    @field_validator('how_long_stay_period')
    def _how_long_stay_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    education_period: float = 0.  # years
    @field_validator('education_period')
    def _education_period(cls, value):
        if (value > 0) and (value < CanadaGeneralConstants.MINIMUM_EDUCATION_PERIOD):
            raise ValueError(
                f'Short term studies cannot be considered.'
                f' Please only provide studies that are over a year.')
        if value > CanadaGeneralConstants.MAXIMUM_EDUCATION_PERIOD:
            raise ValueError('Please only report your LAST education, not the total.')
        return value

    occupation_period: float = 0.   # years
    occupation_period2: float = 0.  # years
    occupation_period3: float = 0.  # years
    @field_validator(
        'occupation_period',
        'occupation_period2',
        'occupation_period3'
    )
    def _occupation_period(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative')
        return value

    applicant_marital_status: str | int | float = 7
    @field_validator('applicant_marital_status')
    def _marital_status(cls, value):
        if isinstance(value, str):
            value = value.lower().strip()
            if value not in CanadaMarriageStatus.get_member_names():
                raise ValueError(
                    f'"{value}" is not valid'
                    f' Please use one of "{CanadaMarriageStatus.get_member_names()}"')
            return value
        elif isinstance(value, int) or isinstance(value, float):
            value = int(value)
            # get Enum values # TODO: use const class or dict
            member_values: List[int] = []
            for member_ in CanadaMarriageStatus._member_names_:
                member_values.append(CanadaMarriageStatus[member_].value)
            if value not in member_values:
                raise ValueError(
                    f'"{value}" is not valid'
                    f' Please use one of "{member_values}"')
            return value
        return value

    previous_country_of_residence_count: int = 0
    @field_validator('previous_country_of_residence_count')
    def _previous_country_of_residence_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        return value

    sibling_foreigner_count: int = 0
    @field_validator('sibling_foreigner_count')
    def _sibling_foreigner_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        if value > CanadaGeneralConstants.MAXIMUM_SIBLING_FOREIGNER_COUNT:
            raise ValueError(
                f'Currently the value cannot be larger than '
                f' "{CanadaGeneralConstants.MAXIMUM_SIBLING_FOREIGNER_COUNT}"')
        return value

    child_mother_father_spouse_foreigner_count: int = 0
    @field_validator('child_mother_father_spouse_foreigner_count')
    def _child_mother_father_spouse_foreigner_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        total_count: int = CanadaGeneralConstants.MAXIMUM_CHILD_COUNT + \
                           CanadaGeneralConstants.MAXIMUM_PARENT_COUNT + \
                           CanadaGeneralConstants.MAXIMUM_SPOUSE_COUNT
        if value > total_count:
            raise ValueError(
                f'Total count of children, parent and spouse cannot be larger'
                f' than 7 (4 child, 2 parent and 1 spouse).')
        return value

    child_accompany: int = 0
    @field_validator('child_accompany')
    def _child_accompany(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        if value > CanadaGeneralConstants.MAXIMUM_CHILD_COUNT:
            raise ValueError(
                f'Number of children accompanying cannot be'
                f' larger than maximum number of children'
                f' (i.e., ={CanadaGeneralConstants.MAXIMUM_CHILD_COUNT})')
        return value

    parent_accompany: int = 0
    @field_validator('parent_accompany')
    def _parent_accompany(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        if value > CanadaGeneralConstants.MAXIMUM_PARENT_COUNT:
            raise ValueError(
                f'Number of parents accompanying cannot be'
                f' larger than maximum number of parents'
                f' (i.e., ={CanadaGeneralConstants.MAXIMUM_PARENT_COUNT})')
        return value

    spouse_accompany: int = 0
    @field_validator('spouse_accompany')
    def _spouse_accompany(cls, value):
        if value < 0:
            raise ValueError(
                f'Value cannot be negative no matter how much you hate your spouse')
        if value > CanadaGeneralConstants.MAXIMUM_SPOUSE_COUNT:
            raise ValueError(
                f'Value cannot be bigger than one (having multiple spouses is ... something!)')
        return value
    sibling_accompany: int = 0

    @field_validator('sibling_accompany')
    def _sibling_accompany(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        if value > CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT:
            raise ValueError(
                f'Number of siblings accompanying cannot be'
                f' larger than maximum number of siblings'
                f' (i.e., ={CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT})')
        return value

    child_average_age: float = 0.  # years
    @field_validator('child_average_age')
    def _child_average_age(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        return value

    child_count: int = 0
    @field_validator('child_count')
    def _child_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        if value > CanadaGeneralConstants.MAXIMUM_CHILD_COUNT:
            raise ValueError(
                f'Currently the value cannot be larger than '
                f' "{CanadaGeneralConstants.MAXIMUM_CHILD_COUNT}"')
        return value
    
    sibling_average_age: int = 0.
    @field_validator('sibling_average_age')
    def _sibling_average_age(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        return value

    sibling_count: int = 0
    @field_validator('sibling_count')
    def _sibling_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        if value > CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT:
            raise ValueError(
                f'Currently the value cannot be larger than '
                f' "{CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT}"')
        return value

    long_distance_child_sibling_count: int = 0

    @field_validator('long_distance_child_sibling_count')
    def _long_distance_child_sibling_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        total_count: int = CanadaGeneralConstants.MAXIMUM_CHILD_COUNT + \
                           CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT
        if value > total_count:
            raise ValueError(
                f'Total count of children and siblings cannot be larger'
                f' than {total_count} '
                f'({CanadaGeneralConstants.MAXIMUM_CHILD_COUNT} children and'
                f' {CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT} siblings).')
        return value

    foreign_living_child_sibling_count: int = 0
    @field_validator('foreign_living_child_sibling_count')
    def _foreign_living_child_sibling_count(cls, value):
        if value < 0:
            raise ValueError('Value cannot be negative.')
        total_count: int = CanadaGeneralConstants.MAXIMUM_CHILD_COUNT + \
                           CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT
        if value > total_count:
            raise ValueError(
                f'Total count of children and siblings cannot be larger'
                f' than {total_count} '
                f'({CanadaGeneralConstants.MAXIMUM_CHILD_COUNT} children and'
                f' {CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT} siblings).')
        return value

    def __init__(self, **data):
        
        # sex
        if 'sex' in data:
            data['sex'] = data['sex'].lower().capitalize()  # female -> Female, ...
        # country_where_applying_country
        if 'country_where_applying_country' in data:
            country_where_applying_country = data['country_where_applying_country']
            def __country_where_applying_country(value: str) -> str:
                value = value.upper()
                if value not in CountryWhereApplying.get_member_names():
                    value = CountryWhereApplying.OTHER.name
                    return value
                if value == CountryWhereApplying.ARMENIA.name:
                    value = 'Armenia'
                    return value
                elif value == CountryWhereApplying.GEORGIA.name:
                    value = 'Georgia'
                    return value
                else:
                    return value
            data['country_where_applying_country'] = __country_where_applying_country(
                value=country_where_applying_country
            )
        
        # occupation_title1, occupation_title2, occupation_title3
        def __occupation_title_x(value: str) -> str:
            value = value.lower()
            if value == OccupationTitle.OTHER.name.lower():
                value = OccupationTitle.OTHER.name
            return value
        if 'occupation_title1' in data:
            data['occupation_title1'] = __occupation_title_x(value=data['occupation_title1'])
        if 'occupation_title2' in data:
            data['occupation_title2'] = __occupation_title_x(value=data['occupation_title2'])
        if 'occupation_title3' in data:
            data['occupation_title3'] = __occupation_title_x(value=data['occupation_title3'])

        # applicant_marital_status
        def __marital_status(value: str) -> float:
            value = value.upper()
            return CanadaMarriageStatus[value].value
        if 'applicant_marital_status' in data:
            data['applicant_marital_status'] = __marital_status(data['applicant_marital_status'])

        # country_where_applying_status
        def __residence_status(value: str) -> float:
            value = value.upper()
            return CanadaResidencyStatus[value].value
        if 'country_where_applying_status' in data:
            data['country_where_applying_status'] = __residence_status(data['country_where_applying_status'])


        # enables adding private methods to Pydantic.BaseModel
        #   see reference: https://github.com/pydantic/pydantic/issues/655
        object.__setattr__(self, 'provided_variables', list(data.keys()))

        super().__init__(**data)
        
        
    class Config:
        from_attributes = True

        @staticmethod
        def json_schema_extra(schema: Dict[str, Any], model: Type['Payload']) -> None:
            # get the json_schema from pydantic
            properties: Dict[Dict[str, str]] = schema.get('properties', {})

            # get all the Field of Payload (i.e., class variables)
            fields: List[str] = list(Payload.model_fields.keys())
            # check if model's fields are subset of documentation fields
            validate_model_fields(model=Payload, fields2docs=payload_fields2docs)
            # traverse through the original properties and add "'doc': documentation" to it
            for field in fields:
                properties[field]['doc'] = payload_fields2docs[field]


class XaiResponse(BaseModel):
    """XAI values for trained model

    Note:
        For more info about XAI and available methods, see :mod:`vizard.xai.shap`. 
    
    """

    xai_overall_score: float
    xai_top_k: Dict[str, float]
    xai_txt_top_k: Dict[str, Tuple[float, str]]


class XaiAggregatedGroupResponse(BaseModel):
    """XAI values grouped and aggregated into categories

    Note:
        This class :class:`vizard.data.constant.FeatureCategories` contains the categories. 
        We use the names of the Enum items.
        For example, :dict:`vizard.data.constant.FEATURE_CATEGORY_TO_FEATURE_NAME_MAP`
        contains the feature names for each categories.

    See Also:
        - XAI and available methods :mod:`vizard.xai.shap`
        - XAI aggregation method :method:`vizard.xai.shap.aggregate_shap_values`
    """

    aggregated_shap_values: Dict[str, float]


class XaiExpandedGroupResponse(BaseModel):
    """XAI values grouped into categories along side features and their XAI values

    Note:
        This class :class:`vizard.data.constant.FeatureCategories` contains the categories. 
        We use the names of the Enum items.
        For example, :dict:`vizard.data.constant.FEATURE_CATEGORY_TO_FEATURE_NAME_MAP`
        contains the feature names for each categories.

    See Also:
        - XAI and available methods :mod:`vizard.xai.shap`
        - XAI sorted group method :method:`vizard.xai.shap.top_k_score`
    """

    grouped_xai_expanded: Dict[str, Dict[str, float]]


class PotentialResponse(BaseModel):
    """Response model for the potential (total XAI) of machine learning model

    Note:
        This is to tell the user how much the user is known to the model
    """
    result: float


class ConstantStatesResponse(BaseModel):
    """Response model for all the constants used throughout APIs

    Note:
        You can find the original constant definitions in :mod:`vizard.data.constant`
    """

    constant_states: Dict[str, List[str]]
