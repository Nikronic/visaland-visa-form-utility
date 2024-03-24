__all__ = ["append_parameter_builder_instances", "logical_questions", "logical_order"]

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

def handle_accompany_questions(next_question: str, is_answered: List[str], answers: Dict) -> str:
    """Combine questions that are about family members accompanying
    Note: 
        in the api we want to ask about the accompany status of family members in one question
        to do that we need to combine the questions that are about family members accompanying
        they are "spouse_accompany", "child_accompany", "parent_accompany" and "sibling_accompany"
        when we know someone is single we dont want to ask about his/her spouse accompanying and
        children accompanying status so we have 3 categories in return we will disuse it there.

    Args:
        next_question (str): next suggested question
        is_answered (List[str]): list of answered questions
        answers (Dict): dictionary of questions and user answers
    Returns:
        str: three things can happen
            1. the question is not related to accompanying and we return next_question without any change.
            2. the question is related to accompanying and we know user is single so we return "singles_accompany_questions". 
            3. the question is related to accompanying and we know user is not single so we return "taken_accompany_questions ".
    """
    pass

def append_parameter_builder_instances(
    suggested: str,
    parameter_builder_instances_names: List[str],
    len_user_answered_params: int,
    len_logically_answered_params: int,
) -> str:
    """Append names of manual parameters to the list of suggested questions



    Args:
        suggested (str): The currently logically suggested question. For more info,
            see :func:`vizard.xai.logical_order`.
        parameter_builder_instances_names (List[str]): List names of the instances of
            :class:`vizard.models.estimators.manual.core.ParameterBuilderBase`. In
            other words, list of names of *manual* parameters.
        len_user_answered_params (int): How many of non-*manual* questions are answered
            by the user. In other words, number of items in the `Payload`.
        len_logically_answered_params (int): How many of questions are logically answered
            given the user answered parameters. For more info, see
            see :func:`vizard.xai.logical_order`.

    Raises:
        ValueError:
            If this method overrides the value of ``suggested`` question that is not
            answered yet by the user or logically.

    Returns:
        str: The next suggested question, which is an item of ``parameter_builder_instances_names``
    """
    # if all non-manual questions are answered
    if len_user_answered_params == len_logically_answered_params:
        # if there are still parameter builder instances not suggested yet
        if len(parameter_builder_instances_names) > 0:
            manual_param_name: str = parameter_builder_instances_names.pop()
            if suggested != "":
                raise ValueError(
                    f'manual parameter "{manual_param_name}" is '
                    f'overriding XAI suggested value "{suggested}"'
                )
            return manual_param_name
    # no suggestion
    return suggested


logical_dict: Dict[str, List[str]] = {
    "spouse_accompany": ["applicant_marital_status"],
    "marriage_period": ["applicant_marital_status"],
    "child_accompany": ["applicant_marital_status", "child_count"],
    "child_count": ["applicant_marital_status"],
}
