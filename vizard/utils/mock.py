import json
import multiprocessing
from itertools import chain, combinations, product
from typing import Any, Dict, List

import httpx

from vizard.data import constant
from vizard.models.estimators.manual import (
    InvitationLetterSenderRelation,
    TravelHistoryRegion,
)

mandatory = ["sex"]
FEATURE_VALUES: Dict[str, List[Any]] = {
    "sex": ["male", "female"],
    "education_field_of_study": ["master", "phd", "unedu"],
    "occupation_title1": ["manager", "specialist", "employee"],
    "refused_entry_or_deport": [True, False],
    "date_of_birth": list(range(20, 61, 10)),
    "marriage_period": list(range(0, 31, 10)),
    "occupation_period": list(range(0, 31, 10)),
    "applicant_marital_status": [5, 7],
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


#####################################
import argparse
import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dvc.api
import mlflow
import numpy as np
import pandas as pd

from vizard.data import functional, preprocessor
from vizard.data.constant import (
    OccupationTitle,
)
from vizard.models import preprocessors, trainers
from vizard.models.estimators.manual import (
    InvitationLetterParameterBuilder,
    InvitationLetterSenderRelation,
    TravelHistoryParameterBuilder,
    TravelHistoryRegion,
)
from vizard.utils import loggers
from vizard.version import VERSION as VIZARD_VERSION
from vizard.xai import FlamlTreeExplainer

# run mlflow tracking server
mlflow.set_tracking_uri(f"http://0.0.0.0:5000")

# data versioning config
PATH = "raw-dataset/all-dev.pkl"  # path to source data, e.g. data.pkl file
REPO = "../visaland-visa-form-utility"
VERSION = "v3.0.0-dev"  # use the latest EDA version (i.e. `vx.x.x-dev`)
# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
data = pd.read_pickle(data_url).drop(columns=["VisaResult"], inplace=False)

# DVC: helper - (for more info see the API that uses these files)
# data file for converting country names to continuous score in "economical" sense
HELPER_PATH_GDP = "raw-dataset/API_NY.GDP.PCAP.CD_DS2_en_xml_v2_4004943.pkl"
HELPER_VERSION_GDP = "v0.1.0-field-GDP"  # use latest using `git tag`
# data file for converting country names to continuous score in "all" possible senses
HELPER_PATH_OVERALL = "raw-dataset/databank-2015-2019.pkl"
HELPER_VERSION_OVERALL = "v0.1.0-field"  # use latest using `git tag`
# gather these for MLFlow track
all_helper_data_info = {
    HELPER_PATH_GDP: HELPER_VERSION_GDP,
    HELPER_PATH_OVERALL: HELPER_VERSION_OVERALL,
}
# data file for converting country names to continuous score in "economical" sense
worldbank_gdp_dataframe = pd.read_pickle(
    dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP)
)
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(
    dataframe=worldbank_gdp_dataframe
)
# data file for converting country names to continuous score in "all" possible senses
worldbank_overall_dataframe = pd.read_pickle(
    dvc.api.get_url(path=HELPER_PATH_OVERALL, repo=REPO, rev=HELPER_VERSION_OVERALL)
)
edu_country_score_preprocessor = (
    preprocessor.EducationCountryScoreDataframePreprocessor(
        dataframe=worldbank_overall_dataframe
    )
)

# configure logging
VERBOSE = logging.DEBUG
MLFLOW_ARTIFACTS_BASE_PATH: Path = Path("artifacts")
if MLFLOW_ARTIFACTS_BASE_PATH.exists():
    shutil.rmtree(MLFLOW_ARTIFACTS_BASE_PATH)
__libs = ["snorkel", "vizard", "flaml"]
logger = loggers.Logger(
    name=__name__,
    level=VERBOSE,
    mlflow_artifacts_base_path=MLFLOW_ARTIFACTS_BASE_PATH,
    libs=__libs,
)

# log experiment configs
MLFLOW_EXPERIMENT_NAME = f"{VIZARD_VERSION}"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.start_run()

logger.info(f"MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}")
logger.info(f"MLflow experiment id: {mlflow.active_run().info.run_id}")

# get mlflow run id for extracting artifacts of the desired run
MLFLOW_RUN_ID = "426deb77881b4d719c2c0d18ce7c36db"
mlflow.log_param("mlflow-trained-run-id", MLFLOW_RUN_ID)

# load fitted preprocessing models
X_CT_NAME = "train_sklearn_column_transfer.pkl"
x_ct_path = mlflow.artifacts.download_artifacts(
    run_id=MLFLOW_RUN_ID,
    artifact_path=f"0/models/{X_CT_NAME}",
    dst_path=f"api/artifacts",
)
with open(x_ct_path, "rb") as f:
    x_ct: preprocessors.ColumnTransformer = pickle.load(f)

# load fitted FLAML AutoML model for prediction
FLAML_AUTOML_NAME = "flaml_automl.pkl"
flaml_automl_path = mlflow.artifacts.download_artifacts(
    run_id=MLFLOW_RUN_ID,
    artifact_path=f"0/models/{FLAML_AUTOML_NAME}",
    dst_path=f"api/artifacts",
)
with open(flaml_automl_path, "rb") as f:
    flaml_automl: trainers.AutoML = pickle.load(f)

feature_names = preprocessors.get_transformed_feature_names(
    column_transformer=x_ct,
    original_columns_names=data.columns.values,
)

# SHAP tree explainer #56
flaml_tree_explainer = FlamlTreeExplainer(
    flaml_model=flaml_automl, feature_names=feature_names, data=None
)

# Create instances of manual parameter insertion
invitation_letter_param = InvitationLetterParameterBuilder()
# Create instances of manual parameter insertion
travel_history_param = TravelHistoryParameterBuilder()
MANUAL_PARAM_NAMES_ANSWERED_DICT: Dict[str, bool] = {
    invitation_letter_param.name: False,
    travel_history_param.name: False,
}

def _predict(is_flagged=False, **kwargs):
    # flag raw data from the user
    if is_flagged:
        logger.debug(f"Raw data from pydantic:")
        logger.debug(f"{kwargs}\n\n")

    # convert api data to model data
    args = list(kwargs.values())
    # convert to dataframe
    x_test = pd.DataFrame(data=[list(args)], columns=data.columns)
    x_test = x_test.astype(data.dtypes)
    x_test = x_test.to_numpy()
    # flag pre transformed data
    if is_flagged:
        logger.debug(f"Preprocessed but not pretransformed data:")
        logger.debug(f"{x_test}\n\n")

    # preprocess test data
    xt_test = x_ct.transform(x_test)
    # flag transformed data
    if is_flagged:
        logger.debug(f"Preprocessed and pretransformed data:")
        logger.debug(f"{xt_test}\n\n")

    # predict
    y_pred = flaml_automl.predict_proba(xt_test)
    label = np.argmax(y_pred)
    y_pred = y_pred[0, label]
    y_pred = y_pred if label == 1 else 1.0 - y_pred
    return y_pred


def predict(features_dict: Dict[str, Any], provided_variables: List[str]):
    # reset manual variable responses
    for k, _ in MANUAL_PARAM_NAMES_ANSWERED_DICT.items():
        MANUAL_PARAM_NAMES_ANSWERED_DICT[k] = False

    # if manual param is answered, make its value True
    for param in provided_variables:
        if param in MANUAL_PARAM_NAMES_ANSWERED_DICT.keys():
            MANUAL_PARAM_NAMES_ANSWERED_DICT[param] = True

    # set response for invitation letter
    invitation_letter_param.set_response(
        response=InvitationLetterSenderRelation(features_dict["invitation_letter"]),
        raw=True,
    )
    # remove invitation letter so preprocessing, transformation, etc works just like before
    if invitation_letter_param.name in features_dict:
        del features_dict[invitation_letter_param.name]
    if invitation_letter_param.name in provided_variables:
        provided_variables.remove(invitation_letter_param.name)

    # set response for travel history
    travel_history_param.set_response(
        response=TravelHistoryRegion(features_dict["travel_history"]),
        raw=True,
    )
    # remove invitation letter so preprocessing, transformation, etc works just like before
    if travel_history_param.name in features_dict:
        del features_dict[travel_history_param.name]
    if travel_history_param.name in provided_variables:
        provided_variables.remove(travel_history_param.name)

    result = _predict(**features_dict)

    # apply invitation letter modification given the response
    result: float = invitation_letter_param.probability_modifier(probability=result)
    # apply travel history modification given the response
    result: float = travel_history_param.probability_modifier(probability=result)

    return result


def para(only):
    results_list = []
    samples_list = []
    sampler = SampleGenerator(FEATURE_VALUES, mandatory, only)
    samples = sampler.sample_maker(FEATURE_VALUES)

    for sample in samples:
        default_feature: Dict[str, Any] = {
            "sex": "string",
            "education_field_of_study": "unedu",
            "occupation_title1": "OTHER",
            "refused_entry_or_deport": False,
            "date_of_birth": 18,
            "marriage_period": 0,
            "occupation_period": 0,
            "applicant_marital_status": "7",
            "child_accompany": 0,
            "parent_accompany": 0,
            "spouse_accompany": 0,
            "sibling_accompany": 0,
            "child_count": 0,
            "invitation_letter": "none",
            "travel_history": "none",
        }
        ###
        for f, v in sample.items():
            default_feature[f] = v
        if "sex" in default_feature:
            default_feature["sex"] = (
                default_feature["sex"].lower().capitalize()
            )  # female -> Female, ...

        def __occupation_title_x(value: str) -> str:
            value = value.lower()
            if value == OccupationTitle.OTHER.name.lower():
                value = OccupationTitle.OTHER.name
            return value

        if "occupation_title1" in default_feature:
            default_feature["occupation_title1"] = __occupation_title_x(
                value=default_feature["occupation_title1"]
            )
        ###
        r = predict(
            features_dict=default_feature,
            provided_variables=list(default_feature.keys()),
        )
        results_list.append(r)
        samples_list.append(default_feature)
    print(
        "number of samples with only subsets of",
        only,
        "->",
        len(samples_list),
        len(samples_list) == len(results_list),
    )  # just to check size and if both are the same size
    import os
    # Save the list to a JSON file
    with open(
        f"{os.getcwd()}/vizard/utils/synthetic_samples/samples_with_subsets_of_only_{only}.json", "w"
    ) as file:  # _samples
        json.dump(samples_list, file, indent=4)
    with open(
        f"{os.getcwd()}/vizard/utils/synthetic_samples/results_with_subsets_of_only_{only}.json", "w"
    ) as file:
        json.dump(results_list, file, indent=4)


print("number of features", len(FEATURE_VALUES))
numbers = list(range(1, 3))
para(only=3)
