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
