from pytest import Parser

from vizard.version import VERSION


def pytest_addoption(parser):
    """A hook to add custom command line options to pytest

    Args:
        parser (:class:`pytest.Parser`):
    """

    parser.addoption("--experiment_name", action="store", default=f"pytest-{VERSION}")
    parser.addoption("--verb", action="store", default="debug")
    parser.addoption(
        "--run_id", action="store", default="ef6d9606e8b64e7eb775ebe79a1b374c"
    )
    parser.addoption("--bind", action="store", default="0.0.0.0")
    parser.addoption("--mlflow_port", action="store", default="5000")
    parser.addoption("--gunicorn_port", action="store", default="8000")
    parser.addoption("--workers", action="store", default="1")


# References:
#   1. https://pytest-with-eric.com/pytest-best-practices/pytest-ini/
