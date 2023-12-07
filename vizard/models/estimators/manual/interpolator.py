__all__ = ["BankBalanceInterpolator"]

from math import isclose
from typing import Any

from vizard.models.estimators.manual.constant import (
    BANK_BALANCE_INPUT_BOUND,
    BANK_BALANCE_STATUS_IMPORTANCE,
    BankBalanceStatus,
)
from vizard.models.estimators.manual.functional import truncated_scaler


class ContinuousInterpolator:
    """The base class for any interpolator used for manual parameters

    Any desired interpolator must extend this class.

    For construction of continuous manual parameters
    :class:`vizard.models.estimators.manual.core.ContinuousParameterBuilderBase` an
    interpolator is required as the possible *responses* (categorical variables does not
    require such interpolation is the value for each category is fixed.). The constant
    values used for construction this interpolator must be defined inside
    :mod:`vizard.models.estimators.manual.constant` module. Note that the final
    constructed manual parameter must reside in :mod:`vizard.models.estimators.manual.core`.

    See Also:

        - :class:`vizard.models.estimators.manual.core.ContinuousParameterBuilderBase`
        - :mod:`vizard.models.estimators.manual.constant`
        - :mod:`vizard.models.estimators.manual.core`

    """

    def __init__(self, **kwargs) -> None:
        pass

    def interpolate(self, x: float) -> float:
        raise NotImplementedError("Please extend this class and implement this method.")


class BankBalanceInterpolator(ContinuousInterpolator):
    """An interpolation for responses of bank balance

    See Also:

        - :class:`vizard.models.estimators.manual.core.BankBalanceContinuousParameterBuilder` that
            contains the main usage of this parameter along side AI model. This interpolator
            class provides the ``responses`` for the aforementioned class.
        - :class:`vizard.models.estimators.manual.constant.BANK_BALANCE_STATUS_IMPORTANCE` that
            contains the lower and upper bounds of the interpolated value.
        - :class:`vizard.models.estimators.manual.constant.BANK_BALANCE_INPUT_BOUND` that
            contains the lower and upper bounds of the input value.
    """

    def __init__(self, **kwargs) -> None:
        self.x_min = BANK_BALANCE_INPUT_BOUND[BankBalanceStatus.LOW]
        self.x_max = BANK_BALANCE_INPUT_BOUND[BankBalanceStatus.HIGH]
        self.target_min = BANK_BALANCE_STATUS_IMPORTANCE[BankBalanceStatus.LOW]
        self.target_max = BANK_BALANCE_STATUS_IMPORTANCE[BankBalanceStatus.HIGH]

    def __check_range(self, x: float) -> None:
        """Check if input ``x`` is respecting its provided range

        Args:
            x (float): Input of interpolation
        """

        if not (isclose(x, self.x_min) or isclose(x, self.x_max)):
            if x > self.x_max or x < self.x_min:
                raise ValueError(f"{x=} is not in range [{self.x_min=}, {self.x_max=}]")

    def interpolate(self, x: float) -> float:
        """A linear function that maps given input to a bound

        Args:
            x (float): Input value to be interpolated

        Returns:
            float: A float in range of ``lower`` and ``upper``.
        """

        self.__check_range(x=x)

        interpolated: float = truncated_scaler(
            x=x,
            x_min_range=self.x_min,
            x_max_range=self.x_max,
            target_min_range=self.target_min,
            target_max_range=self.target_max,
        )

        return interpolated

    def __call__(self, x: float, *args: Any, **kwds: Any) -> Any:
        return self.interpolate(x=x)
