from math import isclose

from pytest import mark

from vizard.models.estimators.manual.interpolator import BankBalanceInterpolator


bank_balance_interpolator = BankBalanceInterpolator()


class TestBankBalanceInterpolator:
    @mark.parametrize(
        argnames=["given_x", "expected_output"],
        argvalues=[(250, -0.013999999999999999), (500, 0.04600000000000001)],
    )
    def test__call__(self, given_x, expected_output):
        output = bank_balance_interpolator(x=given_x)
        assert isclose(expected_output, output)
