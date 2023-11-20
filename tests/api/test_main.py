from fastapi import status
from starlette.testclient import TestClient
from pytest import mark
from api.main import app as vizard_api
from typing import Dict
import math

# the test client for the all tests
client = TestClient(vizard_api)


def test_read_main():
    response = client.get("/")

    expected_response = {"detail": "Not Found"}

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == expected_response


payload_dict: Dict = {
    "sex": "male",
    "country_where_applying_country": "TURKEY",
    "country_where_applying_status": "OTHER",
    "previous_marriage_indicator": False,
    "purpose_of_visit": "tourism",  # 05
    "funds": 8000,
    "contact_relation_to_me": "hotel",
    "education_field_of_study": "unedu",
    "occupation_title1": "OTHER",
    "no_authorized_stay": False,  # 10
    "refused_entry_or_deport": False,
    "previous_apply": False,
    "date_of_birth": 18,
    "country_where_applying_period": 30,
    "marriage_period": 0,  # 15
    "previous_marriage_period": 0,
    "passport_expiry_date_remaining": 3,
    "how_long_stay_period": 30,
    "education_period": 0,
    "occupation_period": 0,  # 20
    "applicant_marital_status": "single",
    "previous_country_of_residence_count": 0,
    "sibling_foreigner_count": 0,
    "child_mother_father_spouse_foreigner_count": 0,
    "child_accompany": 0,  # 25
    "parent_accompany": 0,
    "spouse_accompany": 0,
    "sibling_accompany": 0,
    "child_average_age": 0,
    "child_count": 0,  # 30
    "sibling_average_age": 0,
    "sibling_count": 0,
    "long_distance_child_sibling_count": 0,
    "foreign_living_child_sibling_count": 0,
}

@mark.parametrize(
    argnames=["given", "expected"],
    argvalues=[
        (dict(list(payload_dict.items())[:1]), 0.0026225885130670376),
        (dict(list(payload_dict.items())[:2]), 0.00332688615570846),
        (dict(list(payload_dict.items())[:3]), 0.00332688615570846),
        (dict(list(payload_dict.items())[:4]), 0.007871970202104572),
        (dict(list(payload_dict.items())[:5]), 0.01904925679958386),
        (dict(list(payload_dict.items())[:6]), 0.028250238551028057),
        (dict(list(payload_dict.items())[:7]), 0.07241748836680495),
        (dict(list(payload_dict.items())[:8]), 0.09114285139346825),
        (dict(list(payload_dict.items())[:9]), 0.11561738407372693),
        (dict(list(payload_dict.items())[:10]), 0.11561738407372693),
        (dict(list(payload_dict.items())[:11]), 0.11818193632988706),
        (dict(list(payload_dict.items())[:12]), 0.13373196974102425),
        (dict(list(payload_dict.items())[:13]), 0.2920176526990073),
        (dict(list(payload_dict.items())[:14]), 0.30490183579323554),
        (dict(list(payload_dict.items())[:15]), 0.32792274330461946),
        (dict(list(payload_dict.items())[:16]), 0.3280851883504946),
        (dict(list(payload_dict.items())[:17]), 0.36220177988931135),
        (dict(list(payload_dict.items())[:18]), 0.7002204337233917),
        (dict(list(payload_dict.items())[:19]), 0.7066097581411916),
        (dict(list(payload_dict.items())[:20]), 0.7977812291448462),
        (dict(list(payload_dict.items())[:21]), 0.8035749460781078),
        (dict(list(payload_dict.items())[:22]), 0.8044305709469329),
        (dict(list(payload_dict.items())[:23]), 0.8044305709469329),
        (dict(list(payload_dict.items())[:24]), 0.8075019570914556),
        (dict(list(payload_dict.items())[:25]), 0.8088890372794284),
        (dict(list(payload_dict.items())[:26]), 0.8160160716094819),
        (dict(list(payload_dict.items())[:27]), 0.8178754748164326),
        (dict(list(payload_dict.items())[:28]), 0.8178754748164326),
        (dict(list(payload_dict.items())[:29]), 0.8425011131217108),
        (dict(list(payload_dict.items())[:30]), 0.8453984314637665),
        (dict(list(payload_dict.items())[:31]), 0.8484509792920765),
        (dict(list(payload_dict.items())[:32]), 0.9854033669885174),
        (dict(list(payload_dict.items())[:33]), 0.9880737329330604),
        (dict(list(payload_dict.items())[:34]), 0.9999999),
    ],
)
def test_potential(given: Dict, expected: float):
    http_response = client.post(url="potential/", json=given)
    result: float = dict(http_response.json())["result"]
    assert http_response.status_code == status.HTTP_200_OK
    assert math.isclose(result, expected)
