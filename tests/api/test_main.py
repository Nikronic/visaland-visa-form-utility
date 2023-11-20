# core
from fastapi import status
from starlette.testclient import TestClient

# ours: API
from api.main import app as vizard_api

# the test client for the all tests
client = TestClient(vizard_api)


def test_read_main():
    response = client.get("/")

    expected_response = {"detail": "Not Found"}

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == expected_response


def test_potential():
    response = client.post(url="potential/", json={"sex": "male"})
    expected_response = {"result": 0.0026225885130670376}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response

    response = client.post(
        url="potential/",
        json={"sex": "male", "country_where_applying_country": "TURKEY"},
    )
    expected_response = {"result": 0.00332688615570846}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response

    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER"
        },
    )
    expected_response = {"result": 0.00332688615570846}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False
        },
    )
    expected_response = {"result": 0.007871970202104572}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism"
        },
    )
    expected_response = {"result": 0.01904925679958386}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000
        },
    )
    expected_response = {"result": 0.028250238551028057}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER"
        },
    )
    expected_response = {"result": 0.11561738407372693}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
        },
    )
    expected_response = {"result": 0.11561738407372693}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response
    

    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False
        },
    )
    expected_response = {"result": 0.11818193632988706}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response
    

    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False
        },
    )
    expected_response = {"result": 0.13373196974102425}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18
        },
    )
    expected_response = {"result": 0.2920176526990073}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30
        },
    )
    expected_response = {"result": 0.30490183579323554}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "marriage_period": 0
        },
    )
    expected_response = {"result": 0.32792274330461946}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0
        },
    )
    expected_response = {"result": 0.30506428083911064}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3
        },
    )
    expected_response = {"result": 0.3391808723779274}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30
        },
    )
    expected_response = {"result": 0.6771995262120077}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0
        },
    )
    expected_response = {"result": 0.6835888506298077}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0
        },
    )
    expected_response = {"result": 0.7747603216334622}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single"
        },
    )
    expected_response = {"result": 0.7805540385667238}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0
        },
    )
    expected_response = {"result": 0.781409663435549}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0
        },
    )
    expected_response = {"result": 0.781409663435549}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0
        },
    )
    expected_response = {"result": 0.7844810495800717}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0
        },
    )
    expected_response = {"result": 0.7858681297680445}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0
        },
    )
    expected_response = {"result": 0.792995164098098}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0
        },
    )
    expected_response = {"result": 0.792995164098098}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0
        },
    )
    expected_response = {"result": 0.7948545673050488}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0
        },
    )
    expected_response = {"result": 0.7948545673050488}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_average_age": 0
        },
    )
    expected_response = {"result": 0.819480205610327}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_average_age": 0,
            "child_count": 0
        },
    )
    expected_response = {"result": 0.8223775239523826}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_average_age": 0,
            "child_count": 0,
            "sibling_average_age": 0
        },
    )
    expected_response = {"result": 0.8254300717806925}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_average_age": 0,
            "child_count": 0,
            "sibling_average_age": 0,
            "sibling_count": 0
        },
    )
    expected_response = {"result": 0.9623824594771334}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_average_age": 0,
            "child_count": 0,
            "sibling_average_age": 0,
            "sibling_count": 0,
            "long_distance_child_sibling_count": 0
        },
    )
    expected_response = {"result": 0.9650528254216764}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response


    response = client.post(
        url="potential/",
        json={
            "sex": "male",
            "country_where_applying_country": "TURKEY",
            "country_where_applying_status": "OTHER",
            "previous_marriage_indicator": False,
            "purpose_of_visit": "tourism",
            "funds": 8000,
            "contact_relation_to_me": "hotel",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "no_authorized_stay": False,
            "refused_entry_or_deport": False,
            "previous_apply": False,
            "date_of_birth": 18,
            "country_where_applying_period": 30,
            "previous_marriage_period": 0,
            "passport_expiry_date_remaining": 3,
            "how_long_stay_period": 30,
            "education_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "single",
            "previous_country_of_residence_count": 0,
            "sibling_foreigner_count": 0,
            "child_mother_father_spouse_foreigner_count": 0,
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_average_age": 0,
            "child_count": 0,
            "sibling_average_age": 0,
            "sibling_count": 0,
            "long_distance_child_sibling_count": 0,
            "foreign_living_child_sibling_count": 0
        },
    )
    expected_response = {"result": 0.9769789924886159}
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == expected_response
