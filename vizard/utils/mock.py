from itertools import chain, combinations, product
from typing import Any, Dict, List

from vizard.data import constant
from vizard.models.estimators.manual import (
    InvitationLetterSenderRelation,
    TravelHistoryRegion,
)


class SampleGenerator:
    def __init__(self):
        self.feature_names = [
            "sex",
            "education_field_of_study",
            "occupation_title1",
            "refused_entry_or_deport",
            "date_of_birth",
            "marriage_period",
            "occupation_period",
            "applicant_marital_status",
            "child_accompany",
            "parent_accompany",
            "spouse_accompany",
            "sibling_accompany",
            "child_count",
            "invitation_letter",
            "travel_history",
        ]

        self.feature_values = {
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
            "invitation_letter": list(
                InvitationLetterSenderRelation._value2member_map_.keys()
            ),
            "travel_history": list(TravelHistoryRegion._value2member_map_.keys()),
        }

    @staticmethod
    def powerset(iterable: List) -> List[List]:
        """create a power-set (all possible subsets) from our list
        Args:
            iterable (List): given list of all feature_names to create subsets
        Returns:
            List[List]: a power-set of given feature_names
        """
        s = list(iterable)
        return list(
            chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
        )

    def sample_maker(
        self, feature_names: List = None, feature_values: Dict[str, List] = None
    ) -> List[Dict[str, Any]]:
        """combine all products from product_generator to create all possible samples
        Args:
            feature_names (List): given list of all feature_names to create samples
            feature_values (Dict[str, List]): a dictionary of all acceptable values for each feature -> feature_name:list(acceptable_values)
        Returns:
            List[Dict[str, Any]]: a list of dictionaries each dict is an acceptable fake sample
        """
        if feature_names is None:
            feature_names = self.feature_names  # Set default feature_names

        if feature_values is None:
            feature_values = self.feature_values  # Set default feature_values

        powerset_list = self.powerset(feature_names)
        samples = []
        while powerset_list:
            z = powerset_list.pop()
            sub_dict = self.sub_dict_with_keys(
                z, feature_values
            )  # dict with only keys that are in our sub list
            samples.extend(self.product_generator(sub_dict))
        return samples

    @staticmethod
    def product_generator(dictionary: Dict[str, List]) -> List[Dict[str, Any]]:
        """it gets a dictionary of acceptable values then create product of them
        Args:
            dictionary (Dict[str, List]): a dictionary of all acceptable values for each feature -> feature_name:list(acceptable_values)
        Returns:
            List[Dict[str, Any]]: _description_
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
    def sub_dict_with_keys(
        input_list: List, input_dict: Dict[str, List]
    ) -> Dict[str, List]:
        """take the whole dict and returns sub dict with only items that their key is on our list
        Args:
            input_list (List): list of wanted features to including them from our given dict
            input_dict (Dict[str, List]): a dictionary of all acceptable values for each feature -> feature_name:list(acceptable_values)
        Returns:
            Dict[str, List]: Dict of wanted features and their acceptable values
        """
        return {key: input_dict[key] for key in input_list if key in input_dict}

    def chap(self):
        print(self.feature_names)


feature_values = {
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
