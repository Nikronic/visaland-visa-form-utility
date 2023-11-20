from vizard.xai import utils

is_answered = ["applicant_marital_status"]
answers = {"applicant_marital_status": 7}
question_title = "child_average_age"
correct_order_question = "applicant_marital_status"
correct_answer = (
    [
        "applicant_marital_status",
        "child_count",
        "child_average_age",
        "child_accompany",
        "marriage_period",
        "previous_marriage_period",
        "spouse_accompany",
        "previous_marriage_indicator",
    ],
    {
        "applicant_marital_status": 7,
        "child_count": 0,
        "child_average_age": 0,
        "child_accompany": 0,
        "marriage_period": 0,
        "previous_marriage_period": 0,
        "spouse_accompany": 0,
        "previous_marriage_indicator": False,
    },
)


def test_logical_questions():
    check = utils.logical_questions(is_answered, answers)
    assert check == correct_answer, "test logical questions has been failed"


def test_logical_order():
    check = utils.logical_order(question_title, utils.logical_dict, is_answered)
    assert check == correct_order_question, "test logical order has been failed"
