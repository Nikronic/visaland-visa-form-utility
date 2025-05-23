from math import isclose
from typing import Any, Dict

import pytest
from pytest import mark

from vizard.models.estimators.manual import core

inv_letter_param = core.InvitationLetterParameterBuilder()
travel_hist_param = core.TravelHistoryParameterBuilder()
bank_balance_param = core.BankBalanceContinuousParameterBuilder()


class TestInvitationLetterParameterBuilder:
    @mark.parametrize(
        argnames=[
            "given_potential",
            "given_response",
            "expected_potential",
        ],
        argvalues=[
            (0.0, "f2", 0.5),
            (1.0, "f3", 0.7),
            (0.4, "child", 1.0),
            (0.3, "parent", 0.915),
            (0.5, "none", 0.275),
        ],
    )
    def test_potential_modifier(
        self, given_potential: float, given_response: str, expected_potential: float
    ):
        inv_letter_param.set_response(response=given_response, raw=True)
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
            (1.0, "f3", 0.7),
            (0.3, "child", 1.0),
            (0.3, "parent", 0.915),
            (0.5, "none", 0.275),
        ],
    )
    def test_probability_modifier(
        self, given_probability: float, given_response: str, expected_probability: float
    ):
        inv_letter_param.set_response(response=given_response, raw=True)
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
                    "purpose": 0.7705792574938789,
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
                    "purpose": -0.04984837202086548,
                    "emotional": 0.49196228635250716,
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
                "none",
                {
                    "purpose": -0.19984837202086547,
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
        inv_letter_param.set_response(response=given_response, raw=True)
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
            (0.0, ["schengen_once", "am_ge_tr_az"], 0.55),
            (1.0, "schengen_twice", 1.0),
            (0.4, "jp_kr_af", 0.66),
            (0.4, ["jp_kr_af", "am_ge_tr_az"], 0.71),
            (0.5, "schengen_twice", 1.0),
            (0.9, "am_ge_tr_az", 0.635),
        ],
    )
    def test_potential_modifier(
        self, given_potential: float, given_response: str, expected_potential: float
    ):
        travel_hist_param.set_response(response=given_response, raw=True)
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
            (0.0, ["schengen_once", "am_ge_tr_az"], 0.55),
            (1.0, "schengen_twice", 1.0),
            (0.4, "jp_kr_af", 0.66),
            (0.4, ["jp_kr_af", "am_ge_tr_az"], 0.71),
            (0.5, "schengen_twice", 1.0),
            (0.9, "am_ge_tr_az", 0.635),
        ],
    )
    def test_probability_modifier(
        self, given_probability: float, given_response: str, expected_probability: float
    ):
        travel_hist_param.set_response(response=given_response, raw=True)
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
                    "purpose": 0.8197754861291296,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
            ),
            (
                {
                    "purpose": 0.49196228635250716,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                ["schengen_once", "am_ge_tr_az"],
                {
                    "purpose": 0.8697754861291297,
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
                    "purpose": 0.5138155603389771,
                    "emotional": 0.49196228635250716,
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
                "none",
                {
                    "purpose": -0.23618443966102284,
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
        travel_hist_param.set_response(response=given_response, raw=True)
        new_grouped_xai: float = travel_hist_param.grouped_xai_modifier(
            grouped_xai=given_grouped_xai
        )

        for xai_group_name, _ in new_grouped_xai.items():
            assert isclose(
                new_grouped_xai[xai_group_name], expected_grouped_xai[xai_group_name]
            )


class TestBankBalanceContinuousParameterBuilder:
    @mark.parametrize(
        argnames=[
            "given_potential",
            "given_response",
            "expected_potential",
        ],
        argvalues=[
            (0.0, 100, 0.0),
            (0.0, 500, 0.04600000000000001),
            (1.0, 400, 0.972),
            (0.4, 250, 0.366),
            (0.4, 350, 0.39),
            (0.4, 500, 0.42600000000000005),
        ],
    )
    def test_potential_modifier(
        self, given_potential: float, given_response: str, expected_potential: float
    ):
        bank_balance_param.set_response(response=given_response)
        new_potential: float = bank_balance_param.potential_modifier(given_potential)

        assert isclose(new_potential, expected_potential)

    @mark.parametrize(
        argnames=[
            "given_probability",
            "given_response",
            "expected_probability",
        ],
        argvalues=[
            (0.0, 100, 0.0),
            (0.0, 500, 0.04600000000000001),
            (1.0, 400, 0.972),
            (0.4, 250, 0.366),
            (0.4, 350, 0.39),
            (0.4, 500, 0.42600000000000005),
        ],
    )
    def test_probability_modifier(
        self, given_probability: float, given_response: str, expected_probability: float
    ):
        bank_balance_param.set_response(response=given_response)
        new_probability: float = bank_balance_param.probability_modifier(
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
                    "financial": 0.04314236322943492,
                },
                100,
                {
                    "purpose": 0.49196228635250716,
                    "emotional": -0.3633606764015736,
                    "career": 0.10153467401648415,
                    "financial": -0.00901475493203683,
                },
            ),
            (
                {
                    "purpose": -0.3633606764015736,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                300,
                {
                    "purpose": -0.3633606764015736,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04298524506796317,
                },
            ),
            (
                {
                    "purpose": -0.3633606764015736,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": -0.04314236322943492,
                },
                600,
                {
                    "purpose": -0.3633606764015736,
                    "emotional": 0.49196228635250716,
                    "career": 0.10153467401648415,
                    "financial": 0.029014754932036833,
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
        bank_balance_param.set_response(response=given_response)
        new_grouped_xai: float = bank_balance_param.grouped_xai_modifier(
            grouped_xai=given_grouped_xai
        )

        for xai_group_name, _ in new_grouped_xai.items():
            assert isclose(
                new_grouped_xai[xai_group_name], expected_grouped_xai[xai_group_name]
            )


compose_param = core.ComposeParameterBuilder(
    params=[
        core.InvitationLetterParameterBuilder(),
        core.TravelHistoryParameterBuilder(),
        core.BankBalanceContinuousParameterBuilder(),
    ]
)


class TestComposeParameterBuilder:
    @mark.parametrize(
        argnames=[
            "given_probability",
            "given_il_resp",
            "given_th_resp",
            "given_bb_resp",
            "expected_probability",
        ],
        argvalues=[
            (0.0, "none", "none", 100, 0.0),
            (0.3, "friend", "am_ge_tr_az", 350, 0.22113750000000001),
            (0.3, "f2", "schengen_once", 550, 0.9436375000000001),
        ],
    )
    def test_probability_modifiers(
        self,
        given_probability: float,
        given_il_resp: str,
        given_th_resp: str,
        given_bb_resp: float,
        expected_probability: float,
    ):
        # set responses
        response_dict: Dict[str, Any] = {
            "invitation_letter": given_il_resp,
            "travel_history": given_th_resp,
            "bank_balance": given_bb_resp,
        }
        compose_param.set_responses_for_params(
            responses=response_dict, raw=True, pop=False
        )

        # apply modifiers
        output_probability: float = compose_param.probability_modifiers(
            probability=given_probability
        )

        assert isclose(output_probability, expected_probability)
