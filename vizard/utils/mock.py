import json
import multiprocessing
from itertools import chain, combinations, product
from typing import Any, Dict, List

import httpx

from vizard.data import constant
from vizard.models.estimators.manual import (InvitationLetterSenderRelation,
                                             TravelHistoryRegion)

mandatory = ["sex"]
FEATURE_VALUES: Dict[str, List[Any]] = {
    "sex": ["male", "female"],
    "education_field_of_study": ["master", "phd", "unedu"],
    "occupation_title1": ["manager", "specialist", "employee"],
    # "refused_entry_or_deport": [True, False],
    "date_of_birth": list(range(20, 61, 10)),
    "marriage_period": list(range(0, 31, 10)),
    "occupation_period": list(range(0, 31, 10)),
    "applicant_marital_status": [5, 7],
    "child_accompany": list(range(5)),
    "parent_accompany": list(range(3)),
    "spouse_accompany": list(range(2)),
    "sibling_accompany": list(range(3)),
    "child_count": list(range(5)),
    # "invitation_letter": list(InvitationLetterSenderRelation._value2member_map_.keys()),
    # "travel_history": list(TravelHistoryRegion._value2member_map_.keys()),
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
        self,
        feature_values: Dict[str, List[Any]],
        mandatory_features: Any = None,
        only: Any = None,
    ):
        self.feature_values = feature_values
        self.feature_names = list(self.feature_values.keys())
        self.only = only
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
        only = self.only
        powerset = list(
            chain.from_iterable(
                combinations(iterable, r) for r in range(only - 1, only)
            )
        )
        return powerset

    def _powerset_with_mandatory_features(
        self, iterable: List[str], mandatory_features: Any = None
    ) -> List[List[str]]:
        """sometimes we need some features as mandatory this function keep those features in all of the subsets
        Args:
            iterable (List[str]):  given list of all feature_names to create subsets
            mandatory_features (Any, optional): list of mandatory features

        Returns:
            List[List[str]]: a list of all possible subsets that all of them have our mandatory features
        """
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


url = "http://localhost:9000/predict/"
wanted = "result"


def para(only):
    results_list = []
    samples_list = []
    sampler = SampleGenerator(FEATURE_VALUES, mandatory, only)
    samples = sampler.sample_maker(FEATURE_VALUES)
    with httpx.Client() as client:
        for sample in samples:
            r = client.post(url, json=sample).json()[wanted]
            results_list.append(r)
            samples_list.append(sample)
    print(
        "number of samples with only subsets of",
        only,
        "->",
        len(samples_list),
        len(samples_list) == len(results_list),
    )  # just to check size and if both are the same size

    # Save the list to a JSON file
    with open(
        f"vizard/utils/synthetic_samples/samples_with_subsets_of_only_{only}.json", "w"
    ) as file:  # _samples
        json.dump(samples_list, file, indent=4)
    with open(
        f"vizard/utils/synthetic_samples/results_with_subsets_of_only_{only}.json", "w"
    ) as file:
        json.dump(results_list, file, indent=4)


print("number of features", len(FEATURE_VALUES))
numbers = list(range(1, 5))

with multiprocessing.Pool() as pool:
    results = pool.map(para, numbers)
