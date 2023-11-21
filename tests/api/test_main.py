import json
import math
from pathlib import Path
from typing import Dict, List

from fastapi import status
from pytest import mark
from starlette.testclient import TestClient

from api.main import app as vizard_api

# the test client for the all tests
client = TestClient(vizard_api)


# constants
def __read_json(path: Path) -> None:
    with open(path) as f:
        d = json.load(f)
    return d


# path to the json file containing the full payload
PAYLOAD_JSON = __read_json(path=Path("tests/api/test_main/payload.json"))
# path to the json file containing the step by step response of `potential` endpoint
RESPONSE_POTENTIAL_JSON = __read_json(
    path=Path("tests/api/test_main/response_potential.json")
)

# path to the json file containing the full payload for `predict` endpoint
PAYLOAD_PREDICT_JSON = __read_json(
    path=Path("tests/api/test_main/payload_predict.json")
)
# path to the json file containing the step by step response of `prediction` endpoint
RESPONSE_PREDICT_JSON = __read_json(
    path=Path("tests/api/test_main/response_predict.json")
)

# path to the json file containing the full payload for `grouped_xai` endpoint
PAYLOAD_GROUPED_XAI_JSON = __read_json(
    path=Path("tests/api/test_main/payload_grouped_xai.json")
)
# path to the json file containing the step by step response of `grouped_xai` endpoint
RESPONSE_GROUPED_XAI_JSON = __read_json(
    path=Path("tests/api/test_main/response_grouped_xai.json")
)

# the dictionary of full payload
payload_dict = dict(PAYLOAD_JSON)
# length of payload for generating test cases for pytest parameterization
count: int = len(payload_dict)


def test_read_main():
    response = client.get("/")

    expected_response = {"detail": "Not Found"}

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == expected_response


# path to the json file containing the step by step response of `potential` endpoint
response_potential_list: List[float] = list(dict(RESPONSE_POTENTIAL_JSON)["body"])


@mark.parametrize(
    argnames=["given", "expected"],
    argvalues=[
        (dict(list(payload_dict.items())[: i + 1]), response_potential_list[i])
        for i in range(count)
    ],
)
def test_potential(given: Dict, expected: float):
    http_response = client.post(url="potential/", json=given)
    result: float = dict(http_response.json())["result"]
    assert http_response.status_code == status.HTTP_200_OK
    assert math.isclose(result, expected)


# path to the json file containing the full payload for `predict` endpoint
payload_predict_dict = dict(PAYLOAD_PREDICT_JSON)
# length of payload for generating test cases for pytest parameterization
count: int = len(payload_dict)
# path to the json file containing the step by step response of `predict` endpoint
response_predict_list: List[Dict[str, str | float]] = dict(RESPONSE_PREDICT_JSON)[
    "body"
]


@mark.parametrize(
    argnames=["given", "expected"],
    argvalues=[
        (dict(list(payload_predict_dict.items())[: i + 1]), response_predict_list[i])
        for i in range(count)
    ],
)
def test_predict(given: Dict, expected: Dict[str, str | float]):
    http_response = client.post(url="predict/", json=given)
    assert http_response.status_code == status.HTTP_200_OK

    response_body: Dict[str, str | float] = dict(http_response.json())
    assert math.isclose(response_body["result"], expected["result"])

    assert response_body["next_variable"] == expected["next_variable"]


# path to the json file containing the full payload for `grouped_xai` endpoint
payload_grouped_xai_dict = dict(PAYLOAD_GROUPED_XAI_JSON)
# length of payload for generating test cases for pytest parameterization
count: int = len(payload_grouped_xai_dict)
# path to the json file containing the step by step response of `grouped_xai` endpoint
response_grouped_xai_list: List[Dict[str, float]] = dict(RESPONSE_GROUPED_XAI_JSON)[
    "aggregated_shap_values"
]


@mark.parametrize(
    argnames=["given", "expected"],
    argvalues=[
        (
            dict(list(payload_grouped_xai_dict.items())[: i + 1]),
            response_grouped_xai_list[i],
        )
        for i in range(count)
    ],
)
def test_grouped_xai(given: Dict, expected: Dict[str, float]):
    http_response = client.post(url="grouped_xai/", json=given)
    assert http_response.status_code == status.HTTP_200_OK

    response_body: Dict[str, float] = dict(http_response.json())[
        "aggregated_shap_values"
    ]

    # compare key by key
    for xai_group_ in expected.keys():
        assert math.isclose(response_body[xai_group_], expected[xai_group_])
