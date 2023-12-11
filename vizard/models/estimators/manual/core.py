__all__ = [
    "ParameterBuilderBase",
    "ContinuousParameterBuilderBase",
    "InvitationLetterParameterBuilder",
    "TravelHistoryParameterBuilder",
    "BankBalanceContinuousParameterBuilder",
]

import math
from typing import Any, Callable, Dict, List, Optional

from vizard.data.constant import FeatureCategories
from vizard.models.estimators.manual import constant, functional, interpolator


class ParameterBuilderBase:
    """A base class for applying ``ParameterBuilder` instances as a composable object

    For each parameter that needs to be integrated, this class needs to be extended.

    Using class :class:`vizard.models.estimators.manual.ParameterBuilderBase`
    one can apply different methods as hooks to variables. The specification is for the
    idea that output of a ML modeling (such as XGBoost) requires precise manipulation. Hence,
    all the manipulation required is carried in the definition of classes which extend
    :class:`vizard.models.estimators.manual.ParameterBuilderBase` as "operator"s.

    If we have the following inputs::

    ```python
    vars = {"prob": prob, "pot": pot}
    ```

    Then, when calling this class, we would find a method that matches the name of the input ``vars``
    and then call that method on that those ``vars``. E.g.,::

    ```python
    ParameterBuilder1.prob_modifier(vars["prob"])  # method `prob_modifier` matches `"prob"` variable
    ParameterBuilder1.xai_modifier(None)           # no calls since no match (vars["xai"] is None)

    ParameterBuilder2.prob_modifier(vars["prob"])  # method `prob_modifier` matches `"prob"` variable
    ParameterBuilder2.pot_modifier(vars["pot"])    # method `pot_modifier` matches `"pot"` variable
    ```
    """

    def __init__(
        self,
        name: str,
        responses: Dict[str, float],
        feature_category: FeatureCategories | List[FeatureCategories],
    ) -> None:
        """Initializes a parameter to be built manually

        Args:
            name (str): The name of the parameter (e.g., features in a decision tree)
            responses (Dict[str, float]): Responses or values that this parameter can take with
                their corresponding importance that are normalized (their sum is ``1``). Keys
                are the possible values this parameter takes, and values are the importance
                of each response in range of [0, 1].
            feature_category (:class:`vizard.data.constant.FeatureCategories`): Which category of
                features/parameters this parameter affects. Note that it can take multiple values
                if a list of :class:`vizard.data.constant.FeatureCategories` is provided.
        """
        self.name = name
        self.responses = responses
        self.feature_category = feature_category
        # type check
        self.__type_check()

        # values required for ``_modifier`` methods
        self.response: Optional[str] = None
        self.importance: Optional[float] = None

    def __type_check(self) -> None:
        if not isinstance(self.feature_category, FeatureCategories):
            raise NotImplementedError(
                "Currently, only assignment to a single category is implemented."
            )
        if not isinstance(self.name, str):
            raise ValueError("The name can only be string.")

    def _response_check(self, response: constant.Enum | List[constant.Enum]) -> None:
        """Checks if all instances of response are valid

        Validation is checked upon existence of values in :attr:`responses`

        Args:
            response (Enum | List[Enum]): The list or single instance of response
        """

        if response not in self.responses.keys():
            raise ValueError(
                f"'{response}' is not valid."
                f"Please use one of '{self.responses.keys()}'"
            )

    def _percent_check(self, percent: float) -> None:
        """Checks if the input variable is a percentage in [0, 1]

        Args:
            percent (float): A standardized value
        """

        if not isinstance(percent, float):
            raise ValueError(f"'{percent}' is not a float.")

        # takes care of numerical precision #152
        if not (math.isclose(percent, 0.0) or math.isclose(percent, 1.0)):
            if (percent > 1.0) or (percent < 0.0):
                raise ValueError("'Value should be in '0.0<=value<=1.0'")

    def _grouped_xai_check(self, group: Dict[str, float]) -> None:
        """Checks if the input is a group of percentage in [0, 1]

        Args:
            group (Dict[str, float]): A dictionary where values must be standardized
        """

        for key, value in group.items():
            if not isinstance(key, str):
                raise ValueError("keys must be of type string.")
            # check value to be percentage base
            self._percent_check(abs(value))

    def _check_importance_set(self) -> None:
        """Checks if operators are ready to be used by this class

        raises:
            ValueError: if ``operators`` don't have the ``importance`` attribute set.
                In this case, `operator.set_response` method should be called prior to
                using this method.
        """

        if self.importance is None:
            raise ValueError(
                f"operator must have a value for ``importance``."
                f"`self.set_response` method should be called prior to  using this method."
            )

    def __get_importance(
        self,
        response: constant.Enum | List[constant.Enum],
        raw: bool = False,
    ) -> float:
        """Calculates the importance of the parameter based on the ``response`` given

        Note:
            Method :meth:`vizard.models.estimators.manual.core.set_response` verifies the
            correctness of the ``response`` provided.

        Args:
            response (Enum | List[Enum]): An enum item or list of them representing the possible
                values of this parameter which is a subset of the keys of :attr:`self.responses`.
                The Enum classes reside in :mod:`vizard.models.estimators.manual.constant`.
            raw (bool): Whether to return the raw importance provided initially by the user
                (residing in ``self.responses``) or normalized one if True. Defaults to True.
        """
        total_importance: float = 0.0
        if raw:
            if isinstance(response, constant.Enum):
                total_importance = self.responses[response]
            elif isinstance(response, list):
                total_importance = sum([self.responses[r] for r in response])
                # clip value to a percentage range
                total_importance = self._clip_to_percent(
                    value=total_importance, lower=-1.0, upper=1.0
                )
            else:
                raise ValueError(
                    "importance value cannot be set. Please check your response."
                )
            return total_importance
        raise NotImplementedError("Normalized importance is not yet implemented.")

    def _clip_to_percent(
        self, value: float, lower: float = 0.0, upper: float = 1.0
    ) -> float:
        """Clips a value to the range of [0, 1]

        Note:
            Heuristics used by
            :meth:`vizard.models.estimators.manual.ParameterBuilderBase.potential_modifier`
            or :meth:`vizard.models.estimators.manual.ParameterBuilderBase.potential_modifier`
            might results in scenarios where the output is no longer a percentage. In such
            scenarios, clipping to [0, 1] is desired.

        Args:
            value (float): single number to be clipped
            lower (float, optional): minimum value of clip. Defaults to 0.0.
            upper (float, optional): maximum value of clip. Defaults to 1.0.

        Returns:
            float: A clipped value as a standard percent value
        """

        return max(min(value, upper), lower)  # clip in range of [0, 1]

    def str_to_enum(self, value: str, target_enum: constant.Enum) -> constant.Enum:
        """Takes a string and converts it to Enum type of that string

        When extending this class, this method needs to implemented to take the Enum
        type associated with the extended class.
        For example, for :class:`TravelHistoryParameterBuilder`, the associated enum is
        :class:`vizard.models.estimators.manual.constant.TravelHistoryRegion`.


        Args:
            value (str): The input string to be converted to Enum
            target_enum (Enum): The target Enum representing possible responses

        Returns:
            constant.Enum: The Enum used for computing the importance values
        """

        raise NotImplementedError('Please extend this class and implement this method.')

    def set_response(
        self,
        response: str | List[str],
        raw: bool = False,
    ) -> float:
        """Sets the response to calculate ``self.importance`` used for ``_modifier`` s

        If the input is of type string as the name of Enum items, this method also converts
        them (via :meth:`str_to_enum`) to their Enum item object. Hence user can provide both of type Enum and string.

        Args:
            response (str | List[str]): A string or list of them representing the possible
                values of this parameter. The Enum classes which their name is the possible
                values for responses, reside in :mod:`vizard.models.estimators.manual.constant`.
            raw (bool): Whether to return the raw importance provided initially by the user
                (residing in ``self.responses``) or normalized one if True. Defaults to True.
        Returns:
            float: Returns the calculated importance
        """
        # check if response is valid
        converted_response = response
        if isinstance(response, list):
            converted_response: List[constant.Enum] = []
            for r in response:
                # convert str to Enum
                if isinstance(r, str):
                    r = self.str_to_enum(r)
                    converted_response.append(r)
                self._response_check(response=r)

        elif isinstance(response, constant.Enum):
            self._response_check(response=response)

        elif isinstance(response, str):
            converted_response: constant.Enum = self.str_to_enum(value=response)
            self._response_check(response=converted_response)

        self.response = converted_response
        self.importance = self.__get_importance(response=converted_response, raw=raw)
        # check for range if raw=false
        if not raw:
            self._percent_check(percent=self.importance)
        return self.importance

    def potential_modifier(self, potential: float) -> float:
        """Given an importance (e.g., XAI) recomputes ``potential`` by including this variable

        The value of ``importance`` is proportional to the whole value of ``potential``.
        E.g., if ``importance=0.3``, then when a new ``potential`` is computed,
        this new variable contributes ``%30`` to the overall value of ``potential``.

        Args:
            potential (float): the old potential value without the effect of this variable
        """
        raise NotImplementedError("Please extend this class and implement this method")

    def probability_modifier(self, probability: float) -> float:
        """Given an importance (e.g., XAI) recomputes ``probability`` by including this variable

        The value of ``importance`` is proportional to the whole value of ``probability``.
        E.g., if ``importance=0.3``, then when a new ``probability`` is computed,
        this new variable contributes ``%30`` to the overall value of ``probability``.

        Args:
            probability (float): the old probability value without the effect of this variable
        """
        raise NotImplementedError("Please extend this class and implement this method")

    def grouped_xai_modifier(self, grouped_xai: Dict[str, float]) -> Dict[str, float]:
        """Given an importance (e.g., XAI) recomputes ``grouped_xai`` by including this variable

        The value of ``importance`` is proportional to the value of the key ``self.feature_category``
        (which is one of :class:`vizard.data.constant.FeatureCategories`) in ``grouped_xai``.
        E.g., if ``importance=0.3``, then when a new ``grouped_xai``
        is computed, this new variables contributes ``%30`` to the ``grouped_xai[feature_category]``.

        Args:
            grouped_xai (Dict[str, float]): The old xai values grouped
                by :class:`vizard.data.constant.FeatureCategories`
        Returns:
            Dict[str, float]:
                The new xai values grouped the same way yet the value of
                key ``self.feature_category`` is manipulated by the given ``response``.
        """

        raise NotImplementedError("Please extend this class and implement this method")


class ContinuousParameterBuilderBase(ParameterBuilderBase):
    """A simple extension on top of `ParameterBuilderBase` for continuous parameters

    This class enables us to have an arbitrary
    :class:`vizard.models.estimators.manual.interpolator.ContinuousInterpolator` for obtaining
    the:attr:`importance` values for that continuous param. Any new continuous param should
    extend :class:`vizard.models.estimators.manual.core.ContinuousParameterBuilderBase` class,
    and as before, implement the `_modifier` methods. The bounds or any other *constants* needed
    for this new continuous variable needs to be defined inside the
    :mod:`vizard.models.estimators.manual.constant` module. Also, the interpolator for the importance
    values needs to be defined inside :mod:`vizard.models.estimators.manual.interpolator.` module.

    See Also:
        - :class:`vizard.models.estimators.manual.core.ContinuousParameterBuilderBase`
        - :mod:`vizard.models.estimators.manual.interpolator.`
        - :class:`vizard.models.estimators.manual.interpolator.ContinuousInterpolator`
        - :mod:`vizard.models.estimators.manual.constant`
    """

    def __init__(
        self,
        name: str,
        responses: Callable,
        feature_category: FeatureCategories | List[FeatureCategories],
    ) -> None:
        super().__init__(name, responses, feature_category)

    def _percent_check(self, percent: float) -> None:
        """Checks if the input variable is a percentage in [-1, 1]

        Args:
            percent (float): A standardized value
        """

        if not isinstance(percent, float):
            raise ValueError(f"'{percent}' is not a float.")

        # takes care of numerical precision #152
        if not (math.isclose(percent, -1.0) or math.isclose(percent, 1.0)):
            if (percent > 1.0) or (percent < -1.0):
                raise ValueError("'Value should be in '-1.0<=value<=1.0'")

    def set_response(self, response: float) -> float:
        """Sets the response to calculate ``importance`` used for ``_modifier`` s

        Args:
            response (float): A continuous number as a response that will be passed
                to ``responses`` as a callable that computes the importance dynamically.
        Returns:
            float: Returns the calculated importance
        """

        # check if response is valid
        if not (isinstance(response, float) or isinstance(response, int)):
            raise ValueError(f"'{response}' is not a valid number.")

        self.response = response
        self.importance = self.responses(response)
        # check for range
        self._percent_check(percent=self.importance)
        return self.importance


class InvitationLetterParameterBuilder(ParameterBuilderBase):
    def __init__(self) -> None:
        name: str = "invitation_letter"
        responses: Dict[
            constant.InvitationLetterSenderRelation, float
        ] = constant.INVITATION_LETTER_SENDER_IMPORTANCE
        feature_category: FeatureCategories = FeatureCategories(
            FeatureCategories.PURPOSE
        )

        super().__init__(name, responses, feature_category)

    def str_to_enum(
        self,
        value: str,
        target_enum: constant.Enum = constant.InvitationLetterSenderRelation,
    ) -> constant.InvitationLetterSenderRelation:
        return target_enum(value)

    def potential_modifier(self, potential: float) -> float:
        """Modifies ``potential`` based on given importance

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.potential_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._percent_check(percent=potential)

        new_potential: float = functional.extend_mean(
            percent=potential,
            new_value=self.importance,
            shift=self.responses[constant.InvitationLetterSenderRelation.BASE],
        )

        # `potential` is a percent value
        new_potential = self._clip_to_percent(value=new_potential)

        return new_potential

    def probability_modifier(self, probability: float) -> float:
        """Modifies ``probability`` based on given importance

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.probability_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._percent_check(percent=probability)

        new_probability: float = functional.extend_mean(
            percent=probability,
            new_value=self.importance,
            shift=self.responses[constant.InvitationLetterSenderRelation.BASE],
        )

        # `new_probability` is a percent value
        new_probability = self._clip_to_percent(value=new_probability)

        return new_probability

    def grouped_xai_modifier(self, grouped_xai: Dict[str, float]) -> Dict[str, float]:
        """Modifies ``grouped_xai`` based on given importance

        Note:
            This operation is not ``inplace``.

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.grouped_xai_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._grouped_xai_check(group=grouped_xai)

        # get the group assigned in parameter builder
        xai_group: str = FeatureCategories(self.feature_category).name
        # create a new dictionary to prevent inplace operation
        new_grouped_xai: Dict[str, float] = {}
        # TODO: implement list of feature_category

        # update the key that matches `feature_category`
        for key, value in grouped_xai.items():
            if key == xai_group:
                value = functional.extend_mean(
                    percent=value,
                    new_value=self.importance,
                    shift=self.responses[constant.InvitationLetterSenderRelation.BASE],
                )
            new_grouped_xai[key] = value
        return new_grouped_xai


class TravelHistoryParameterBuilder(ParameterBuilderBase):
    def __init__(self) -> None:
        name: str = "travel_history"
        responses: Dict[
            constant.TravelHistoryRegion, float
        ] = constant.TRAVEL_HISTORY_REGION_IMPORTANCE
        feature_category: FeatureCategories = FeatureCategories(
            FeatureCategories.PURPOSE
        )

        super().__init__(name, responses, feature_category)

    def str_to_enum(
        self, value: str, target_enum: constant.Enum = constant.TravelHistoryRegion
    ) -> constant.TravelHistoryRegion:
        return target_enum(value)

    def potential_modifier(self, potential: float) -> float:
        """Modifies ``potential`` based on given importance

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.potential_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._percent_check(percent=potential)

        new_potential: float = functional.extend_mean(
            percent=potential,
            new_value=self.importance,
            shift=self.responses[constant.TravelHistoryRegion.BASE],
        )

        # `potential` is a percent value
        new_potential = self._clip_to_percent(value=new_potential)

        return new_potential

    def probability_modifier(self, probability: float) -> float:
        """Modifies ``probability`` based on given importance

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.probability_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._percent_check(percent=probability)

        new_probability: float = functional.extend_mean(
            percent=probability,
            new_value=self.importance,
            shift=self.responses[constant.TravelHistoryRegion.BASE],
        )

        # `new_probability` is a percent value
        new_probability = self._clip_to_percent(value=new_probability)

        return new_probability

    def grouped_xai_modifier(self, grouped_xai: Dict[str, float]) -> Dict[str, float]:
        """Modifies ``grouped_xai`` based on given importance

        Note:
            This operation is not ``inplace``.

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.grouped_xai_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._grouped_xai_check(group=grouped_xai)

        # get the group assigned in parameter builder
        xai_group: str = FeatureCategories(self.feature_category).name
        # create a new dictionary to prevent inplace operation
        new_grouped_xai: Dict[str, float] = {}
        # TODO: implement list of feature_category

        # update the key that matches `feature_category`
        for key, value in grouped_xai.items():
            if key == xai_group:
                value = functional.extend_mean(
                    percent=value,
                    new_value=self.importance,
                    shift=self.responses[constant.TravelHistoryRegion.BASE],
                )
            new_grouped_xai[key] = value
        return new_grouped_xai


class BankBalanceContinuousParameterBuilder(ContinuousParameterBuilderBase):
    """Manual continuous parameter for `bank_balance`

    See Also:

        - :class:`vizard.models.estimators.manual.interpolator.BankBalanceInterpolator`
        - :mod:`vizard.models.estimators.manual.constant.BankBalanceStatus`
        - :dict:`vizard.models.estimators.manual.constant.BANK_BALANCE_STATUS_IMPORTANCE`
        - :dict:`vizard.models.estimators.manual.constant.BANK_BALANCE_INPUT_BOUND`

    """

    def __init__(self) -> None:
        name: str = "bank_balance"
        responses: Callable = interpolator.BankBalanceInterpolator()
        feature_category: FeatureCategories = FeatureCategories(
            FeatureCategories.FINANCIAL
        )

        super().__init__(name, responses, feature_category)

    def potential_modifier(self, potential: float) -> float:
        """Modifies ``potential`` based on given importance

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.potential_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._percent_check(percent=potential)

        new_potential: float = functional.extend_mean(
            percent=potential,
            new_value=self.importance,
            shift=constant.BANK_BALANCE_STATUS_IMPORTANCE[
                constant.BankBalanceStatus.BASE
            ],
        )

        # `potential` is a percent value
        new_potential = self._clip_to_percent(value=new_potential)

        return new_potential

    def probability_modifier(self, probability: float) -> float:
        """Modifies ``probability`` based on given importance

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.probability_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._percent_check(percent=probability)

        new_probability: float = functional.extend_mean(
            percent=probability,
            new_value=self.importance,
            shift=constant.BANK_BALANCE_STATUS_IMPORTANCE[
                constant.BankBalanceStatus.BASE
            ],
        )

        # `new_probability` is a percent value
        new_probability = self._clip_to_percent(value=new_probability)

        return new_probability

    def grouped_xai_modifier(self, grouped_xai: Dict[str, float]) -> Dict[str, float]:
        """Modifies ``grouped_xai`` based on given importance

        Note:
            This operation is not ``inplace``.

        See Also:
            Base method :meth:`vizard.models.estimators.manual.ParameterBuilderBase.grouped_xai_modifier`
        """
        # check if response is provided
        self._check_importance_set()
        # check input is valid
        self._grouped_xai_check(group=grouped_xai)

        # get the group assigned in parameter builder
        xai_group: str = FeatureCategories(self.feature_category).name
        # create a new dictionary to prevent inplace operation
        new_grouped_xai: Dict[str, float] = {}
        # TODO: implement list of feature_category

        # update the key that matches `feature_category`
        for key, value in grouped_xai.items():
            if key == xai_group:
                value = functional.extend_mean(
                    percent=value,
                    new_value=self.importance,
                    shift=constant.BANK_BALANCE_STATUS_IMPORTANCE[
                        constant.BankBalanceStatus.BASE
                    ],
                )
            new_grouped_xai[key] = value
        return new_grouped_xai
