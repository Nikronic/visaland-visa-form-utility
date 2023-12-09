from itertools import chain, combinations, product
from typing import Any, Dict, List

from vizard.data import constant
from vizard.models.estimators.manual import (InvitationLetterSenderRelation,
                                             TravelHistoryRegion)

mandatory = ["sex"]
FEATURE_VALUES: Dict[str, List[Any]] = {
    "sex": ["male", "female"],
    "education_field_of_study": constant.EducationFieldOfStudy.get_member_names(),
    "occupation_title1": constant.OccupationTitle.get_member_names(),
    "refused_entry_or_deport": [True, False],
    "date_of_birth": list(range(18, 55)),
    "marriage_period": list(range(30)),
    "occupation_period": list(range(30)),
    "applicant_marital_status": [3, 5, 7, 8],
    "child_accompany": list(range(5)),
    "parent_accompany": list(range(3)),
    "spouse_accompany": list(range(2)),
    "sibling_accompany": list(range(3)),
    "child_count": list(range(5)),
    "invitation_letter": list(InvitationLetterSenderRelation._value2member_map_.keys()),
    "travel_history": list(TravelHistoryRegion._value2member_map_.keys()),
}


class SampleGenerator:
    """Generates samples using predefined feature names and values.

    Attributes:
        feature_names (List[str]): List of feature names.
        feature_values (Dict[str, List[Any]]): Dictionary of feature values.
    Note:
        This class helps generate artificial samples to test our trained model.
        It utilizes predefined features and their possible values to create mock
        data for analysis.
    """

    # TODO : check if it works well with no mandatory_features

    def __init__(
        self, feature_values: Dict[str, List[Any]], mandatory_features: Any = None
    ):
        self.feature_values = feature_values
        self.feature_names = list(self.feature_values.keys())
        if mandatory_features is None:
            self.mandatory_features = []
        else:
            self.mandatory_features = mandatory_features

    def _powerset(self, iterable: List[str]) -> List[List[str]]:
        """create a power-set (all possible subsets) from our list
        Args:
            iterable (List[str]): given list of all feature_names to create subsets
        Returns:
            List[List[str]]: a power-set of given feature_names
        """
        powerset = list(
            chain.from_iterable(
                combinations(iterable, r) for r in range(len(iterable) + 1)
            )
        )
        return powerset

    def _powerset_with_mandatory_features(
        self, iterable: List[str], mandatory_features: Any = None
    ) -> List[List[str]]:
        if mandatory_features is None or mandatory_features == []:
            powerset = self._powerset(list_without_mandatory_features)
            if {} in powerset:
                powerset.remove({})
            return powerset
        else:
            list_without_mandatory_features = [
                features
                for features in iterable
                if features not in self.mandatory_features
            ]  # remove mandatory items from list
            powerset = self._powerset(list_without_mandatory_features)

            customize_powerset = (
                []
            )  # all subset has mandatory items if mandatory is given
            for single_tuple in powerset:
                customize_powerset.append(list(single_tuple))  # change tuples to list
            for subset in customize_powerset:
                subset.extend(
                    self.mandatory_features
                )  # add mandatory_features to all subsets

            return customize_powerset

    def sample_maker(
        self,
        feature_names: List[str] = None,
        feature_values: Dict[str, List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """combine all products from product_generator to create all possible samples
        Args:
            feature_names (List[str]): given list of all feature_names to create samples
            feature_values (Dict[str, List[Any]]): a dictionary of all acceptable values for each feature
        Returns:
            List[Dict[str, Any]]: a list of dictionaries each dict is an acceptable fake sample
        """
        if feature_names is None:
            feature_names = self.feature_names  # Set default feature_names

        if feature_values is None:
            feature_values = self.feature_values  # Set default feature_values

        powerset_list = self._powerset_with_mandatory_features(
            feature_names, self.mandatory_features
        )  # TODO: check this line
        samples = []
        while powerset_list:
            subset = powerset_list.pop()
            sub_dict = self._sub_dict_with_keys(
                subset, feature_values
            )  # dict with only keys that are in our sub list
            samples.extend(self._product_generator(sub_dict))
        return samples

    @staticmethod
    def _product_generator(
        dictionary: Dict[str, List[Any]]
    ) -> List[Dict[str, List[Any]]]:
        """it gets a dictionary of acceptable values then create product of them
        Args:
            dictionary (Dict[str, List[Any]]): a dictionary of all acceptable values for each feature
        Returns:
            List[Dict[str, List[Any]]]: product from possible values that their feature is on given list
        """
        keys = list(dictionary.keys())  # Convert keys to a list
        value_lists = [dictionary[key] for key in keys]

        # Generate all possible combinations of values
        all_combinations = list(product(*value_lists))

        # Create dictionaries for each combination
        result = []
        for combination in all_combinations:
            sample_dict = {keys[i]: combination[i] for i in range(len(keys))}
            result.append(sample_dict)

        return result

    @staticmethod
    def _sub_dict_with_keys(
        input_list: List[str], input_dict: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """take the whole dict and returns sub dict with only items that their key is on our list
        Args:
            input_list (List[str]): list of wanted features to including them from our given dict
            input_dict (Dict[str, List[[Any]]]): a dictionary of all acceptable values for each feature
        Returns:
            Dict[str, List[[Any]]]: Dict of wanted features and their acceptable values
        """
        return {key: input_dict[key] for key in input_list if key in input_dict}
