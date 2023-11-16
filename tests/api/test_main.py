# core
from fastapi import status
from starlette.testclient import TestClient

# ours: API
from api.main import app as vizard_api

# the test client for the all tests
client = TestClient(vizard_api)


def test_read_main():
    response = client.get('/')

    expected_response = {
        'detail': 'Not Found'
    }

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == expected_response
