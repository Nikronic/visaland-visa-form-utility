from vizard.data import constant
from vizard.xai import utils


def test_logical_questions():
    correct_answer = (
        [
            "applicant_marital_status",
            "child_count",
            "child_accompany",
            "marriage_period",
            "spouse_accompany",
        ],
        {
            "applicant_marital_status": constant.CanadaMarriageStatus.SINGLE.value,
            "child_count": 0,
            "child_accompany": 0,
            "marriage_period": 0,
            "spouse_accompany": 0,
        },
    )
    is_answered = ["applicant_marital_status"]
    answers = {"applicant_marital_status": constant.CanadaMarriageStatus.SINGLE.value}
    check = utils.logical_questions(is_answered, answers)
    assert check == correct_answer, "test logical questions has been failed"


def test_logical_order():
    is_answered = ["applicant_marital_status"]
    question_title = "child_accompany"
    correct_order_question = "child_count"
    check = utils.logical_order(question_title, utils.logical_dict, is_answered)
    assert check == correct_order_question, "test logical order has been failed"
