from typing import Dict, List, Tuple, Any
from vizard.data.constant import FEATURE_CATEGORY_TO_FEATURE_NAME_MAP, FeatureCategories
import numpy as np


def get_top_k_idx(sample: np.ndarray, k: int) -> np.ndarray:
    """Extracts top-k indices of an array

    Args:
        sample (:class:`numpy.ndarray`): A single instance :class:`numpy.ndarray`
        k (int): Number of items to return

    Return:
        :class:`numpy.ndarray`: List of top-k indices
    """
    top_k_idx: np.ndarray = np.argpartition(sample, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(sample[top_k_idx])][::-1]
    return top_k_idx


def get_top_k(
    sample: np.ndarray, k: int, absolute: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts top-k items in an numpy array conditioned on sign of values

    Args:
        sample (:class:`numpy.ndarray`): A :class:`numpy.ndarray` array
        k (int): Number of items to return
        absolute (bool, optional): Wether or not to consider
            the sign of values in computing top-k. If ``True``,
            then absolute values used and vice versa. Defaults to True.

    Raises:
        NotImplementedError: If ``sample`` contains multiple samples (rows).

    Returns:
        Tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`]:
            Top-k values of array ``sample`` and their indices in a tuple.
    """
    # if single instance is given
    if (sample.ndim == 1) or ((sample.ndim == 2) and (sample.shape[0] == 1)):
        sample = sample.flatten()

        # if absolute top k should be chosen
        top_k_idx: np.ndarray
        if absolute:
            # absolute top k
            sample_abs: np.ndarray = np.abs(sample)
            top_k_idx = get_top_k_idx(sample=sample_abs, k=k)
        else:
            # signed top k
            top_k_idx = get_top_k_idx(sample=sample, k=k)

        # top k values with their signs
        top_k: np.ndarray = sample[top_k_idx]
        return (top_k, top_k_idx)
    else:
        raise NotImplementedError(
            "This method is only available for" " single-instance numpy arrays. (yet)"
        )


def xai_threshold_to_text(xai_value: float, threshold: float = 0.0) -> str:
    """Converts a XAI value to a negative/positive text by thresholding

    Args:
        xai_value (float): XAI value to be interpreted
        threshold (float): XAI threshold

    Returns:
        str: a negative/positive string satisfying ``threshold``
    """

    if xai_value >= threshold:
        if xai_value < 0.5:
            return "خوب است"
        else:
            return "عالی است"
    else:
        return "بد است"


def xai_to_text(
    xai_feature_values: Dict[str, float], feature_to_keyword_mapping: Dict[str, str]
) -> Dict[str, Tuple[float, str]]:
    """Takes XAI values for features and generates basic textual descriptions

    XAI values are computed using `SHAP` (see :class:`vizard.xai.shap`). Then, I
    use a simple dictionary that has a basic general statement for each feature.
    Via a simple thresholding, I generate textual descriptions for each feature.
    e.g., I take ``"P3.DOV.PrpsRow1.HLS.Period": 1.5474060773849487`` and generate
    ``"P3.DOV.PrpsRow1.HLS.Period": [1.5474060773849487, "مدت زمان اقامت خوب است"]``


    See Also:

        - XAI module: :mod:`vizard.xai`
        - SHAP module: :mod:`vizard.xai.shap`
        - A mapping of features to basic text: ``vizard.data.constant.FEATURE_NAME_TO_TEXT_MAP``
        - XAI value thresholding method: :meth:`vizard.xai.core.xai_threshold_to_text`

    Args:
        xai_feature_values (Dict[str, float]): XAI values for features
        feature_to_keyword_mapping (Dict[str, str]): Mapping from feature to keyword

    Returns:
        Dict[str, float, str]: XAI values for features with text
    """
    xai_txt_top_k: Dict[str, Tuple[float, str]] = {}
    for _feature_name, _feature_xai_value in xai_feature_values.items():
        xai_txt_top_k[_feature_name] = (
            _feature_xai_value,
            (
                f"{feature_to_keyword_mapping[_feature_name]}"
                f" {xai_threshold_to_text(xai_value=_feature_xai_value, threshold=0.)}"
            ),
        )

    return xai_txt_top_k


def xai_category_texter(
    xai_feature_values: Dict[str, float],
    feature_to_keyword_mapping: Dict[str, str],
    answers_tuple: tuple[List[str], Dict[str, Any]],
):
    """Takes XAI values for features and generates basic textual descriptions"""
    is_answered = answers_tuple[0]
    answers = answers_tuple[1]
    # Mapping from long names to short names
    name_map = {
        "P1.PD.DOBYear.Period": "date_of_birth",
        "p1.SecA.ParAccomp.Count": "parent_accompany",
        "P1.MS.SecA.DateOfMarr.Period": "marriage_period",
        "p1.SecA.Sps.SpsAccomp.Count": "spouse_accompany",
        "p1.SecB.Chd.X.ChdRel.ChdCount": "child_count",
        "p1.SecA.App.ChdMStatus": "applicant_marital_status",
        "p1.SecB.Chd.X.ChdAccomp.Count": "child_accompany",
        "p1.SecC.Chd.X.ChdAccomp.Count": "sibling_accompany",
        "P3.Occ.OccRow1.Period": "occupation_period",
        "P3.Edu.Edu_Row1.FieldOfStudy": "education_field_of_study",
        "P1.PD.Sex.Sex": "sex",
        "P3.refuseDeport": "refused_entry_or_deport",
        "P3.Occ.OccRow1.Occ.Occ": "occupation_title1",
        "P3.DOV.PrpsRow1.Funds.Funds": "bank_balance",
        # manually added
        "travel_history": "travel_history",
        "invitation_letter": "invitation_letter",
    }

    categorical_features_map = {
        "P3.Edu.Edu_Row1.FieldOfStudy_master": "education_field_of_study",
        "P3.Edu.Edu_Row1.FieldOfStudy_phd": "education_field_of_study",
        "P3.Edu.Edu_Row1.FieldOfStudy_apprentice": "education_field_of_study",
        "P3.Edu.Edu_Row1.FieldOfStudy_diploma": "education_field_of_study",
        "P3.Edu.Edu_Row1.FieldOfStudy_unedu": "education_field_of_study",
        "P3.Edu.Edu_Row1.FieldOfStudy_bachelor": "education_field_of_study",
        "P3.Occ.OccRow1.Occ.Occ_specialist": "occupation_title1",
        "P3.Occ.OccRow1.Occ.Occ_OTHER": "occupation_title1",
        "P3.Occ.OccRow1.Occ.Occ_student": "occupation_title1",
        "P3.Occ.OccRow1.Occ.Occ_manager": "occupation_title1",
        "P3.Occ.OccRow1.Occ.Occ_housewife": "occupation_title1",
        "P3.Occ.OccRow1.Occ.Occ_retired": "occupation_title1",
        "P3.Occ.OccRow1.Occ.Occ_employee": "occupation_title1",
        "P3.refuseDeport_True": "refused_entry_or_deport",
        "P3.refuseDeport_False": "refused_entry_or_deport",
        "P1.PD.Sex.Sex_Female": "sex",
        "P1.PD.Sex.Sex_Male": "sex",
    }
    # manually added features
    xai_include_manual_assigns = xai_feature_values
    if answers["bank_balance"] > 10000:
        xai_include_manual_assigns["P3.DOV.PrpsRow1.Funds.Funds"] = 0.05
    else:
        xai_include_manual_assigns["P3.DOV.PrpsRow1.Funds.Funds"] = -0.08
    if answers["travel_history"] != ["none"]:
        xai_include_manual_assigns["travel_history"] = 0.5
    else:
        xai_include_manual_assigns["travel_history"] = -0.2
    if answers["invitation_letter"] == "":
        xai_include_manual_assigns["invitation_letter"] = 0.5
    else:
        xai_include_manual_assigns["invitation_letter"] = -0.2

    filtered_list = filter_elements(
        xai_feature_values.keys(), name_map, is_answered, answers
    )[0]
    response_explain = {
        key.name: [] for key in FEATURE_CATEGORY_TO_FEATURE_NAME_MAP.keys()
    }

    for _feature_name, _feature_xai_value in xai_include_manual_assigns.items():
        if _feature_name in filtered_list:
            for (
                _feature_category,
                _feature_name_list,
            ) in FEATURE_CATEGORY_TO_FEATURE_NAME_MAP.items():
                if _feature_name in _feature_name_list:
                    if _feature_name in name_map:
                        name = name_map[_feature_name]
                    elif _feature_name in categorical_features_map:
                        name = categorical_features_map[_feature_name]
                    else:
                        name = _feature_name
                    response_explain[_feature_category.name].append(
                        {
                            "name": name,
                            "value": _feature_xai_value,
                            "txt": f"{feature_to_keyword_mapping[_feature_name]} {xai_threshold_to_text(xai_value=_feature_xai_value, threshold=0.)}",
                        }
                    )

    return response_explain


def filter_elements(input_list, name_mapping, answered_questions, answers):

    # Reverse mapping from short names to long names
    reverse_mapping = {v: k for k, v in name_mapping.items()}

    # List of all categorical possible questions
    categorical_questions = [
        "education_field_of_study",
        "sex",
        "occupation_title1",
        "refused_entry_or_deport",
    ]

    # Dictionary to store answered questions and their answers
    answered_dict = {}

    # List to store filtered elements
    filtered_elements = []

    # List of all categorical possible answers
    categorical_possible_answers = [
        "P3.Edu.Edu_Row1.FieldOfStudy_master",
        "P3.Edu.Edu_Row1.FieldOfStudy_phd",
        "P3.Edu.Edu_Row1.FieldOfStudy_apprentice",
        "P3.Edu.Edu_Row1.FieldOfStudy_diploma",
        "P3.Edu.Edu_Row1.FieldOfStudy_unedu",
        "P3.Edu.Edu_Row1.FieldOfStudy_bachelor",
        "P3.Occ.OccRow1.Occ.Occ_specialist",
        "P3.Occ.OccRow1.Occ.Occ_OTHER",
        "P3.Occ.OccRow1.Occ.Occ_student",
        "P3.Occ.OccRow1.Occ.Occ_manager",
        "P3.Occ.OccRow1.Occ.Occ_housewife",
        "P3.Occ.OccRow1.Occ.Occ_retired",
        "P3.Occ.OccRow1.Occ.Occ_employee",
        "P3.refuseDeport_True",
        "P3.refuseDeport_False",
        "P1.PD.Sex.Sex_Female",
        "P1.PD.Sex.Sex_Male",
    ]

    # to make the value of refused_entry_or_deport to be capitalized
    answers["refused_entry_or_deport"] = str(
        answers["refused_entry_or_deport"]
    ).capitalize()

    for question in categorical_questions:
        if question in answered_questions:
            answered_dict[question] = answers[question]

    # Filter all_possible_answers to only include elements that match the answered questions and their answers
    filtered_elements = [
        answer
        for answer in categorical_possible_answers
        if any(
            reverse_mapping[question] in answer and answers[question] in answer
            for question in answered_questions
        )
    ]

    # Create a new list with only the items that should be kept
    filtered_input_list = [
        item
        for item in input_list
        if (
            item not in categorical_possible_answers
            and name_mapping[item]
            in answered_questions  # Remove the items that are not in answered
        )
        or item in filtered_elements
    ]

    return filtered_input_list, filtered_elements
