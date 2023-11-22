from math import isclose

from pytest import mark

from vizard.models.estimators.manual import constant, core

inv_letter_param = core.InvitationLetterParameterBuilder()


class TestInvitationLetterParameterBuilder:
    @mark.parametrize(
        argnames=[
            "given_potential",
            "given_response",
            "expected_potential",
        ],
        argvalues=[
            (0.0, "f2", 0.5),
            (1.0, "f3", 1.0),
            (0.4, "child", 1.0),
            (0.5, "parent", 0.875),
        ],
    )
    def test_potential_modifier(
        self, given_potential: float, given_response: str, expected_potential: float
    ):
        inv_letter_param.set_response(
            response=constant.InvitationLetterSenderRelation(given_response), raw=True
        )
        new_potential: float = inv_letter_param.potential_modifier(given_potential)

        assert isclose(new_potential, expected_potential)

    @mark.parametrize(
        argnames=[
            "given_probability",
            "given_response",
            "expected_probability",
        ],
        argvalues=[
            (0.0, "f2", 0.5),
            (1.0, "f3", 1.0),
            (0.3, "child", 1.0),
            (0.7, "parent", 0.925),
        ],
    )
    def test_probability_modifier(
        self, given_probability: float, given_response: str, expected_probability: float
    ):
        inv_letter_param.set_response(
            response=constant.InvitationLetterSenderRelation(given_response), raw=True
        )
        new_probability: float = inv_letter_param.probability_modifier(
            given_probability
        )

        assert isclose(new_probability, expected_probability)
