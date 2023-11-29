__all__ = [
    "PredictionResponse",
    "Payload",
    "ConstantStatesResponse",
    "XaiResponse",
    "XaiAggregatedGroupResponse",
    "PotentialResponse",
]

import json
from typing import Any, Dict, List, Tuple, Type

import pydantic
from pydantic import ConfigDict, field_validator
from pydantic.fields import FieldInfo

from vizard.data.constant import (
    CanadaGeneralConstants,
    CanadaMarriageStatus,
    EducationFieldOfStudy,
    FeatureCategories,
    OccupationTitle,
    Sex,
)
from vizard.models.estimators.manual import (
    InvitationLetterSenderRelation,
    TravelHistoryRegion,
)


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


def validate_model_fields(model: BaseModel, fields2docs: Dict[str, str]) -> None:
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
        raise ValueError("Not all model fields are covered by ``fields2docs``.")


payload_fields2docs: Dict[str, str] = {
    "sex": 'The sex of the user. Input must be one of ["Female", "Male"]',
    "education_field_of_study": 'Study level of the user. Input must be one of ["phd", "master", "bachelor", "diploma", "apprentice", "unedu"]',
    "occupation_title1": 'Occupation of the user. Input must be one of ["manager", "specialist", "employee", "retired", "student", "OTHER"]',
    "refused_entry_or_deport": "Has the user denied entry or deported? Input must be `true` or `false`",
    "date_of_birth": "The age of the person. Input must be a single float in years larger than 18.",
    "applicant_marital_status": 'The marital status of the user. Input must be one of ["single", "married", "divorced", "widowed", "unknown"]',
    "marriage_period": "How long the user has been married. Input must be a single float in years (zero if not married).",
    "occupation_period": "How long the user has been doing the `occupation_title1` job. Input must be a single float in years.",
    "child_accompany": "How many of the users children are accompanying the user. Input must be a single integer.",
    "parent_accompany": "Are the parents of the user accompanying him/her? Input must be a single integer between 0 to 2. 0 means none, 2 means both father and mother.",
    "spouse_accompany": "Is the spouse of the user accompanying him/her? Input must be either 0 or 1 (you can use `true` or `false`).",
    "sibling_accompany": "How many of the siblings are accompanying the user. Input must be a integer (0 if none accompanying).",
    "child_count": "How many children the user has. Input must be a integer (0 if no children).",
    "invitation_letter": "The relation of the person sending you the invitation letter. Input must be one of ['child', 'sibling', 'parent', 'f2', 'f3', 'friend', 'spouse', 'pro_unrelated', 'pro_related', 'none']",
    "travel_history": "The history of travels based on customized regions. Input must be one of ['schengen_once', schengen_twice', 'us_uk_au', 'jp_kr_af', 'br_sg_th_my_ru', 'ae_om_qa', 'am_ge_tr_az', 'none']",
}
"""A dictionary of input payload and their documentation for OpenAPI docs

The keys are the fields of :class:`Payload` and values are the description (documentation)
for each field. This dictionary is used to extend the `json_schema` generated by ``Pydantic``
to include these documentations.

See Also:
    class `Config` inside :class:`Payload`.
"""


class Payload(BaseModel):
    def _json_schema_extra(schema: Dict[str, Any], model: Type["Payload"]) -> None:
        # get the json_schema from pydantic
        properties: Dict[Dict[str, str]] = schema.get("properties", {})

        # get all the Field of Payload (i.e., class variables)
        fields: List[str] = list(Payload.model_fields.keys())
        # check if model's fields are subset of documentation fields
        validate_model_fields(model=Payload, fields2docs=payload_fields2docs)
        # traverse through the original properties and add "'doc': documentation" to it
        for field in fields:
            properties[field]["doc"] = payload_fields2docs[field]

    model_config = ConfigDict(
        from_attributes=True, json_schema_extra=_json_schema_extra
    )

    # required for having dynamic request body for populating ``next_variable`` of `ResponseModel`
    __slots__ = ("provided_variables",)

    sex: str

    @field_validator("sex")
    def _sex(cls, value):
        if value not in Sex.get_member_names():
            raise ValueError(
                f'"{value}" is not valid.'
                f' Only "{Sex.get_member_names()}" are available.'
            )
        return value

    education_field_of_study: str = "unedu"

    @field_validator("education_field_of_study")
    def _education_field_of_study(cls, value):
        value = value.lower().strip()
        if value not in EducationFieldOfStudy.get_member_names():
            raise ValueError(
                f'"{value}" is not valid'
                f' Please use one of "{EducationFieldOfStudy.get_member_names()}"'
            )
        return value

    occupation_title1: str = "OTHER"

    @field_validator("occupation_title1")
    def _occupation_title(cls, value):
        if value not in OccupationTitle.get_member_names():
            raise ValueError(
                f'"{value}" is not valid'
                f' Please use one of "{OccupationTitle.get_member_names()}"'
            )
        return value

    refused_entry_or_deport: bool = False

    date_of_birth: float = 18.0

    @field_validator("date_of_birth")
    def _date_of_birth(cls, value):
        if value < CanadaGeneralConstants.MINIMUM_AGE:
            raise ValueError("This service only accepts adults")
        return value

    marriage_period: float = 0.0  # years

    @field_validator("marriage_period")
    def _marriage_period(cls, value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        return value

    occupation_period: float = 0.0  # years

    @field_validator("occupation_period")
    def _occupation_period(cls, value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        return value

    applicant_marital_status: str | int | float = 7

    @field_validator("applicant_marital_status")
    def _marital_status(cls, value):
        if isinstance(value, str) and (not value.isnumeric()):
            value = value.lower().strip()
            if value not in CanadaMarriageStatus.get_member_names():
                raise ValueError(
                    f'"{value}" is not valid'
                    f' Please use one of "{CanadaMarriageStatus.get_member_names()}"'
                )
            return CanadaMarriageStatus[value.upper()].value
        elif isinstance(value, int) or isinstance(value, float) or value.isnumeric():
            value = int(value)
            # get Enum values # TODO: use const class or dict
            member_values: List[int] = []
            for member_ in CanadaMarriageStatus._member_names_:
                member_values.append(CanadaMarriageStatus[member_].value)
            if value not in member_values:
                raise ValueError(
                    f'"{value}" is not valid' f' Please use one of "{member_values}"'
                )
            return value
        return value

    child_accompany: int = 0

    @field_validator("child_accompany")
    def _child_accompany(cls, value):
        if value < 0:
            raise ValueError("Value cannot be negative.")
        if value > CanadaGeneralConstants.MAXIMUM_CHILD_COUNT:
            raise ValueError(
                f"Number of children accompanying cannot be"
                f" larger than maximum number of children"
                f" (i.e., ={CanadaGeneralConstants.MAXIMUM_CHILD_COUNT})"
            )
        return value

    parent_accompany: int = 0

    @field_validator("parent_accompany")
    def _parent_accompany(cls, value):
        if value < 0:
            raise ValueError("Value cannot be negative.")
        if value > CanadaGeneralConstants.MAXIMUM_PARENT_COUNT:
            raise ValueError(
                f"Number of parents accompanying cannot be"
                f" larger than maximum number of parents"
                f" (i.e., ={CanadaGeneralConstants.MAXIMUM_PARENT_COUNT})"
            )
        return value

    spouse_accompany: int = 0

    @field_validator("spouse_accompany")
    def _spouse_accompany(cls, value):
        if value < 0:
            raise ValueError(
                f"Value cannot be negative no matter how much you hate your spouse"
            )
        if value > CanadaGeneralConstants.MAXIMUM_SPOUSE_COUNT:
            raise ValueError(
                f"Value cannot be bigger than one (having multiple spouses is ... something!)"
            )
        return value

    sibling_accompany: int = 0

    @field_validator("sibling_accompany")
    def _sibling_accompany(cls, value):
        if value < 0:
            raise ValueError("Value cannot be negative.")
        if value > CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT:
            raise ValueError(
                f"Number of siblings accompanying cannot be"
                f" larger than maximum number of siblings"
                f" (i.e., ={CanadaGeneralConstants.MAXIMUM_SIBLING_COUNT})"
            )
        return value

    child_count: int = 0

    @field_validator("child_count")
    def _child_count(cls, value):
        if value < 0:
            raise ValueError("Value cannot be negative.")
        if value > CanadaGeneralConstants.MAXIMUM_CHILD_COUNT:
            raise ValueError(
                f"Currently the value cannot be larger than "
                f' "{CanadaGeneralConstants.MAXIMUM_CHILD_COUNT}"'
            )
        return value

    invitation_letter: str = InvitationLetterSenderRelation.NONE.value

    @field_validator("invitation_letter")
    def _invitation_letter(cls, value):
        if value.lower() not in InvitationLetterSenderRelation._value2member_map_:
            raise ValueError(
                f"'{value}' is not valid"
                f" Please use one of '{list(InvitationLetterSenderRelation._value2member_map_.keys())}'"
            )
        return value

    travel_history: str = TravelHistoryRegion.NONE.value

    @field_validator("travel_history")
    def _travel_history(cls, value):
        if value.lower() not in TravelHistoryRegion._value2member_map_:
            raise ValueError(
                f"'{value}' is not valid"
                f" Please use one of '{list(TravelHistoryRegion._value2member_map_.keys())}'"
            )
        return value

    def __init__(self, **data):
        # sex
        if "sex" in data:
            data["sex"] = data["sex"].lower().capitalize()  # female -> Female, ...

        # occupation_title1, occupation_title2, occupation_title3
        def __occupation_title_x(value: str) -> str:
            value = value.lower()
            if value == OccupationTitle.OTHER.name.lower():
                value = OccupationTitle.OTHER.name
            return value

        if "occupation_title1" in data:
            data["occupation_title1"] = __occupation_title_x(
                value=data["occupation_title1"]
            )

        # enables adding private methods to Pydantic.BaseModel
        #   see reference: https://github.com/pydantic/pydantic/issues/655
        object.__setattr__(self, "provided_variables", list(data.keys()))

        super().__init__(**data)


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
