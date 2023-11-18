def logical_questions(is_answered: list, answers):
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
        "previous_marriage_indicator" in is_answered
        and answers["previous_marriage_indicator"] == False
    ) or (
        "applicant_marital_status" in is_answered
        and answers["applicant_marital_status"] == 7
    ):
        answers["child_count"] = 0
        answers["child_average_age"] = 0
        answers["child_accompany"] = 0
        answers["marriage_period"] = 0
        answers["previous_marriage_period"] = 0
        answers["spouse_accompany"] = 0
        answers["applicant_marital_status"] = 7
        answers["previous_marriage_indicator"] = False

        extend_list = [
            "child_count",
            "child_average_age",
            "child_accompany",
            "marriage_period",
            "previous_marriage_period",
            "spouse_accompany",
            "applicant_marital_status",
            "previous_marriage_indicator",
        ]

        is_answered.extend(extend_list)
        is_answered = list(dict.fromkeys(is_answered))  # Remove Duplicates

    if "sibling_count" in is_answered and answers["sibling_count"] == 0:
        answers["sibling_average_age"] = 0
        answers["sibling_accompany"] = 0
        answers["sibling_foreigner_count"] = 0

        extend_list = [
            "sibling_average_age",
            "sibling_accompany",
            "sibling_foreigner_count",
        ]
        is_answered.extend(extend_list)
        is_answered = list(dict.fromkeys(is_answered))  # Remove Duplicates

    if "child_count" in is_answered and answers["child_count"] == 0:
        answers["child_average_age"] = 0
        answers["child_accompany"] = 0

        extend_list = ["child_average_age", "child_accompany"]
        is_answered.extend(extend_list)
        is_answered = list(dict.fromkeys(is_answered))  # Remove Duplicates

    return is_answered, answers
