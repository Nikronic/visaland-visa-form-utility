from typing import Dict, List, Tuple

from vizard.data import constant


def logical_questions(is_answered: List, answers: Dict) -> Tuple:
    """check the logical sense of answer to automatically answer some other questions

    Note:
        like if someone is single he isn't going aboard with his wife

    Args:
        is_answered (List): a list of answered questions
        answers (Dict): a dictionary of questions and user answers

    Returns:
        tuple:
            the first argument is a list of answered questions after we automatically answer some questions
            and the second one is a dictionary of questions and user answers after we automatically answer some questions
    """

    if (
        "applicant_marital_status" in is_answered
        and answers["applicant_marital_status"]
        == constant.CanadaMarriageStatus.SINGLE.value
    ):
        answers["child_count"] = 0
        answers["child_accompany"] = 0
        answers["marriage_period"] = 0
        answers["spouse_accompany"] = 0

        extend_list = [
            "child_count",
            "child_accompany",
            "marriage_period",
            "spouse_accompany",
        ]

        is_answered.extend(extend_list)

    if "child_count" in is_answered and answers["child_count"] == 0:
        answers["child_accompany"] = 0

        extend_list = ["child_accompany"]
        is_answered.extend(extend_list)

    # Remove Duplicates in is_answered
    is_answered = list(dict.fromkeys(is_answered))

    return is_answered, answers


def logical_order(
    question_title: str, logical_dict: Dict[str, List[str]], is_answered: List[str]
) -> str:
    """check logical order of questions
    Note:
        there is a suggested question that has highest information gain
        we check is it logical to ask that or we should go to another one
        like asking average age of kids before person in question said he/she has any
    Args:
        question_title (str): given title of the suggested question
        logical_dict (Dict[str, List[str]]): given dict that represent logical order
        is_answered (List): a list of answered questions

    Returns:
        str: logical question that we should ask with the given suggestion
    """
    if question_title in logical_dict and not (
        logical_dict[question_title][-1] in is_answered
    ):
        for item in logical_dict[question_title]:
            if not item in is_answered:
                return item
    else:
        return question_title


logical_dict: Dict[str, List[str]] = {
    "spouse_accompany": ["applicant_marital_status"],
    "marriage_period": ["applicant_marital_status"],
    "child_accompany": ["applicant_marital_status", "child_count"],
    "child_count": ["applicant_marital_status"],
    "sibling_accompany": ["sibling_count"],
}
