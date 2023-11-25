from math import isclose
from typing import Dict

from pytest import mark

from vizard.models.estimators.manual import constant, core

inv_letter_param = core.InvitationLetterParameterBuilder()
travel_hist_param = core.TravelHistoryParameterBuilder()


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

    @mark.parametrize(
        argnames=[
            "given_grouped_xai",
            "given_response",
            "expected_grouped_xai",
        ],
        argvalues=[
            (
                {
                    "purpose": 0.49196228635250716,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                "f2",
                {
                    "purpose": 0.7459811431762535,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
            ),
            (
                {
                    "purpose": -0.3633606764015736,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                "f3",
                {
                    "purpose": -0.15885657494133756,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
            ),
        ],
    )
    def test_grouped_xai_modifier(
        self,
        given_grouped_xai: Dict[str, float],
        given_response: str,
        expected_grouped_xai: Dict[str, float],
    ):
        inv_letter_param.set_response(
            response=constant.InvitationLetterSenderRelation(given_response), raw=True
        )
        new_grouped_xai: float = inv_letter_param.grouped_xai_modifier(
            grouped_xai=given_grouped_xai
        )

        for xai_group_name, _ in new_grouped_xai.items():
            assert isclose(
                new_grouped_xai[xai_group_name], expected_grouped_xai[xai_group_name]
            )


class TestTravelHistoryParameterBuilder:
    @mark.parametrize(
        argnames=[
            "given_potential",
            "given_response",
            "expected_potential",
        ],
        argvalues=[
            (0.0, "schengen_once", 0.5),
            (1.0, "schengen_twice", 1.0),
            (0.4, "jp_kr_af", 0.64),
            (0.5, "schengen_twice", 0.875),
        ],
    )
    def test_potential_modifier(
        self, given_potential: float, given_response: str, expected_potential: float
    ):
        travel_hist_param.set_response(
            response=constant.TravelHistoryRegion(given_response), raw=True
        )
        new_potential: float = travel_hist_param.potential_modifier(given_potential)

        assert isclose(new_potential, expected_potential)

    @mark.parametrize(
        argnames=[
            "given_probability",
            "given_response",
            "expected_probability",
        ],
        argvalues=[
            (0.0, "schengen_once", 0.5),
            (1.0, "schengen_twice", 1.0),
            (0.3, "jp_kr_af", 0.58),
            (0.7, "schengen_twice", 0.925),
        ],
    )
    def test_probability_modifier(
        self, given_probability: float, given_response: str, expected_probability: float
    ):
        travel_hist_param.set_response(
            response=constant.TravelHistoryRegion(given_response), raw=True
        )
        new_probability: float = travel_hist_param.probability_modifier(
            given_probability
        )

        assert isclose(new_probability, expected_probability)

    @mark.parametrize(
        argnames=[
            "given_grouped_xai",
            "given_response",
            "expected_grouped_xai",
        ],
        argvalues=[
            (
                {
                    "purpose": 0.49196228635250716,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                "schengen_once",
                {
                    "purpose": 0.7459811431762535,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
            ),
            (
                {
                    "purpose": -0.3633606764015736,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                "schengen_twice",
                {
                    "purpose": 0.6591598308996066,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
            ),
        ],
    )
    def test_grouped_xai_modifier(
        self,
        given_grouped_xai: Dict[str, float],
        given_response: str,
        expected_grouped_xai: Dict[str, float],
    ):
        travel_hist_param.set_response(
            response=constant.TravelHistoryRegion(given_response), raw=True
        )
        new_grouped_xai: float = travel_hist_param.grouped_xai_modifier(
            grouped_xai=given_grouped_xai
        )

        for xai_group_name, _ in new_grouped_xai.items():
            assert isclose(
                new_grouped_xai[xai_group_name], expected_grouped_xai[xai_group_name]
            )
