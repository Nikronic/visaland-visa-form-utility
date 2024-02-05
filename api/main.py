import argparse
import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dvc.api
import fastapi
import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from vizard.api import apps as api_apps
from vizard.api import database as api_database
from vizard.api import models as api_models
from vizard.data import functional, preprocessor
from vizard.data.constant import (
    FEATURE_CATEGORY_TO_FEATURE_NAME_MAP,
    FEATURE_NAME_TO_TEXT_MAP,
    CanadaMarriageStatus,
    EducationFieldOfStudy,
    FeatureCategories,
    OccupationTitle,
)
from vizard.models import preprocessors, trainers
from vizard.models.estimators.manual import (
    BankBalanceContinuousParameterBuilder,
    ComposeParameterBuilder,
    InvitationLetterParameterBuilder,
    InvitationLetterSenderRelation,
    TravelHistoryParameterBuilder,
    TravelHistoryRegion,
)
from vizard.seduce.models.name_generator import RecordGenerator
from vizard.utils import loggers
from vizard.version import VERSION as VIZARD_VERSION
from vizard.xai import FlamlTreeExplainer, utils, xai_to_text

# argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--experiment_name",
    type=str,
    help="mlflow experiment name for logging",
    default="",
    required=True,
)
parser.add_argument(
    "-d",
    "--verbose",
    type=str,
    help="logging verbosity level.",
    choices=["debug", "info"],
    default="info",
    required=True,
)
parser.add_argument(
    "-r",
    "--run_id",
    type=str,
    help="MLflow run ID to extract artifacts (weights, models, etc)",
    required=True,
)
parser.add_argument(
    "-b",
    "--bind",
    type=str,
    help="ip address of host",
    default="0.0.0.0",
    required=True,
)
parser.add_argument(
    "-m",
    "--mlflow_port",
    type=int,
    help="port of mlflow tracking",
    default=5000,
    required=True,
)
parser.add_argument(
    "-g",
    "--gunicorn_port",
    type=int,
    help="port used for creating gunicorn",
    default=8000,
    required=True,
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    help="number of works used by gunicorn",
    default=1,
    required=True,
)
args = parser.parse_args()

# run mlflow tracking server
mlflow.set_tracking_uri(f"http://{args.bind}:{args.mlflow_port}")

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
VERBOSE = logging.DEBUG if args.verbose == "debug" else logging.INFO
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
if args.experiment_name == "":
    MLFLOW_EXPERIMENT_NAME = f"{VIZARD_VERSION}"
else:
    MLFLOW_EXPERIMENT_NAME = f"{args.experiment_name} - {VIZARD_VERSION}"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.start_run()

logger.info(f"MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}")
logger.info(f"MLflow experiment id: {mlflow.active_run().info.run_id}")

# get mlflow run id for extracting artifacts of the desired run
MLFLOW_RUN_ID = args.run_id
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
travel_history_param = TravelHistoryParameterBuilder()
bank_balance_continuous_param = BankBalanceContinuousParameterBuilder()
param_composer = ComposeParameterBuilder(
    params=[
        invitation_letter_param,
        travel_history_param,
        bank_balance_continuous_param,
    ]
)

# instantiate fast api app
app = fastapi.FastAPI(
    title="Vizard",
    summary="Visa chance AI assistant",
    version=VIZARD_VERSION,
)

# fastapi cross origin
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _xai(**kwargs):
    # convert api data to model data
    args = list(kwargs.values())
    # convert to dataframe
    x_test = pd.DataFrame(data=[list(args)], columns=data.columns)
    x_test = x_test.astype(data.dtypes)
    x_test = x_test.to_numpy()
    # preprocess test data
    xt_test = x_ct.transform(x_test)
    return xt_test


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


def _potential(**kwargs):
    # 1. create a one-to-one mapping from payload variables to data columns
    payload_variables: List = list(kwargs.keys())
    column_names_to_payload: Dict[str, str] = {
        column_name: payload_v
        for column_name, payload_v in zip(list(data.columns.values), payload_variables)
    }

    # 2. create a one-to-many mapping from data columns to transformed feature names
    payload_to_transformed_feature_names: Dict[str, List[str]] = {}

    def _get_indices(sublist: List[str], superlist: List[str]) -> List[int]:
        """Finds the index of strings of B in A where strings in B have similar initial chars as strings in A

        Note:
            This is used for finding the indices of features that are related to a specific topic.

        Args:
            sublist (List[str]): List of strings as subset of ``superlist``
            superlist (List[str]): List of strings where are shortened versions of strings in
                ``sublist``.

        Returns:
            List[int]: List of indices of strings of ``sublist`` in ``superlist``
        """

        return [
            i
            for item in sublist
            for i, s_item in enumerate(superlist)
            if s_item.startswith(item)
        ]

    for _og_feature in list(data.columns.values):
        # get indices of transformed features resulted from original features
        features_idx = _get_indices(sublist=[_og_feature], superlist=feature_names)
        payload_to_transformed_feature_names[column_names_to_payload[_og_feature]] = [
            feature_names[_feature_idx] for _feature_idx in features_idx
        ]

    # 3. get feature names' xai values
    xai_input: np.ndarray = _xai(**kwargs)
    # compute xai values for the sample
    xai_top_k: Dict[str, float] = flaml_tree_explainer.top_k_score(
        sample=xai_input, k=-1
    )

    # 4. provide a one-to-one mapping from payload variables to xai values
    # assign the aggregated xai value of transformed features to payload variables
    payload_to_xai: Dict[str, float] = {}
    for _payload_v, _tf_names in payload_to_transformed_feature_names.items():
        total_xai_for_tf_names: List[int] = [
            xai_top_k[_tf_name] for _tf_name in _tf_names
        ]
        payload_to_xai[_payload_v] = np.sum(np.absolute(total_xai_for_tf_names)).item()

    return payload_to_xai


@app.post("/potential/", response_model=api_models.PotentialResponse)
async def potential(features: api_models.Payload):
    # calculate the potential: some of abs xai values for given variables
    try:
        # TODO: this method is broken reason conflict features.provided_variables and is_answered
        features_dict: Dict[str, Any] = features.model_dump()
        provided_variables: List[str] = features.provided_variables

        # set response for manual params `invitation_letter`, `travel_history`, and `bank_balance`
        param_composer.set_responses_for_params(
            responses={
                invitation_letter_param.name: features.invitation_letter,
                travel_history_param.name: features.travel_history,
                bank_balance_continuous_param.name: features.bank_balance,
            },
            raw=True,
            pop=True,
            pop_containers=[features_dict, provided_variables],
        )

        payload_to_xai = _potential(**features_dict)

        # compute dictionary of payloads provided and their xai values
        provided_payload: Dict[str, float] = dict(
            (k, payload_to_xai[k]) for k in provided_variables if k in payload_to_xai
        )
        potential_by_xai_raw: float = np.sum(np.abs(list(provided_payload.values())))
        # total XAI values as the denominator (normalizer)
        total_xai: float = np.sum(np.abs(list(payload_to_xai.values())))
        # normalize to 0-1 for percentage
        potential_by_xai_normalized: float = potential_by_xai_raw / total_xai

        # apply manual params modification given the response
        potential_by_xai_normalized: float = param_composer.potential_modifiers(
            potential=potential_by_xai_normalized
        )

        # TEMP: hardcoded small value to prevent 1.0 from happening just for fun
        FUN_EPSILON: float = 1e-7
        return {"result": potential_by_xai_normalized - FUN_EPSILON}

    except Exception as error:
        e = sys.exc_info()[1]
        logger.exception(e)
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/predict/", response_model=api_models.PredictionResponse)
async def predict(
    features: api_models.Payload,
):
    try:
        features_dict: Dict[str, Any] = features.model_dump()
        provided_variables: List[str] = features.provided_variables

        # if manual param is answered, consume it
        param_composer.consume_params(params_names=provided_variables)

        # set response for manual params `invitation_letter`, `travel_history`, and `bank_balance`
        param_composer.set_responses_for_params(
            responses={
                invitation_letter_param.name: features.invitation_letter,
                travel_history_param.name: features.travel_history,
                bank_balance_continuous_param.name: features.bank_balance,
            },
            raw=True,
            pop=True,
            pop_containers=[features_dict, provided_variables],
        )

        logic_answers_implanted = utils.logical_questions(
            provided_variables, features_dict
        )
        is_answered = logic_answers_implanted[
            0
        ]  # TODO: conflict features.provided_variables
        given_answers = logic_answers_implanted[1]
        result = _predict(**given_answers)

        # apply manual params modification given the response
        result: float = param_composer.probability_modifiers(probability=result)

        # get the next question by suggesting the variable with highest XAI value
        payload_to_xai: Dict[str, float] = _potential(**features_dict)

        # remove variables that are in the payload (already answered)
        for provided_variable_ in is_answered:
            del payload_to_xai[provided_variable_]

        next_suggested_variable: str = ""
        next_logical_variable: str = next_suggested_variable
        if payload_to_xai:
            next_suggested_variable = max(
                payload_to_xai, key=lambda xai_value: np.abs(payload_to_xai[xai_value])
            )
            next_logical_variable = utils.logical_order(
                next_suggested_variable, utils.logical_dict, is_answered
            )

        # add instances of manual variables' names as a next suggested variable
        next_logical_variable = utils.append_parameter_builder_instances(
            suggested=next_logical_variable,
            parameter_builder_instances_names=list(
                k for k, v in param_composer.consumption_status_dict.items() if not v
            ),
            len_user_answered_params=len(features_dict),
            len_logically_answered_params=len(is_answered),
        )

        logger.info("Inference finished")
        return {"result": result, "next_variable": next_logical_variable}
    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/flag/", response_model=api_models.PredictionResponse)
async def flag(
    features: api_models.Payload,
):
    is_flagged: bool = True
    # create new instance of mlflow artifact
    logger.create_artifact_instance()

    try:
        result = _predict(**features.model_dump())

        # if need to be flagged, save as artifact
        if is_flagged:
            logger.debug(f"Features Pydantic type passed to the main endpoints:")
            logger.debug(f"{features}\n\n")
            logger.debug(f"Features dict type passed to the main endpoints:")
            logger.debug(f"{features.__dict__}\n\n")
            logger.info(f"artifacts saved in MLflow artifacts directory.")
            mlflow.log_artifacts(MLFLOW_ARTIFACTS_BASE_PATH)

        return {
            "result": result,
        }
    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/xai", response_model=api_models.XaiResponse)
async def xai(features: api_models.Payload, k: int = 5):
    # validate sample
    sample = _xai(**features.model_dump())

    # compute xai values for the sample
    xai_overall_score: float = flaml_tree_explainer.overall_score(sample=sample)
    xai_top_k: Dict[str, float] = flaml_tree_explainer.top_k_score(sample=sample, k=k)

    # TODO: cannot retrieve value for transformed (let's say categorical)
    # for i, (k, v) in enumerate(xai_top_k.items()):
    # print(f'idx={i} => feat={k}, val={sample[0, i]}, xai={v}\n')

    # dict of {feature_name, xai value, textual description}
    xai_txt_top_k: Dict[str, Tuple[float, str]] = xai_to_text(
        xai_feature_values=xai_top_k,
        feature_to_keyword_mapping=FEATURE_NAME_TO_TEXT_MAP,
    )

    return {
        "xai_overall_score": xai_overall_score,
        "xai_top_k": xai_top_k,
        "xai_txt_top_k": xai_txt_top_k,
    }


@app.post("/grouped_xai_expanded", response_model=api_models.XaiExpandedGroupResponse)
async def grouped_xai_expanded(features: api_models.Payload):
    features_dict: Dict[str, Any] = features.model_dump()

    # set response for manual params `invitation_letter`, `travel_history`, and `bank_balance`
    param_composer.set_responses_for_params(
        responses={
            invitation_letter_param.name: features.invitation_letter,
            travel_history_param.name: features.travel_history,
            bank_balance_continuous_param.name: features.bank_balance,
        },
        raw=True,
        pop=True,
        pop_containers=[features_dict],
    )

    # validate sample
    sample = _xai(**features_dict)

    # compute xai values for the sample
    xai_top_k: Dict[str, float] = flaml_tree_explainer.top_k_score(sample=sample, k=-1)

    # grouped xai with all features included (X):
    #   get all the features of that xai group: A
    #   get all the xai values of that xai group: B
    #   create a dict where keys are xai groups, and value are
    #       list of C=Ai:Bi

    grouped_xai_expanded: Dict[FeatureCategories, Dict[str, float]] = {}

    # a one-to-many mapping from data columns to transformed feature names
    og_feature_to_transformed_feature_names: Dict[str, List[str]] = {}

    def _get_indices(sublist: List[str], superlist: List[str]) -> List[int]:
        """Finds the index of strings of B in A where strings in B have similar initial chars as strings in A

        Note:
            This is used for finding the indices of features that are related to a specific topic.

        Args:
            sublist (List[str]): List of strings as subset of ``superlist``
            superlist (List[str]): List of strings where are shortened versions of strings in
                ``sublist``.

        Returns:
            List[int]: List of indices of strings of ``sublist`` in ``superlist``
        """

        return [
            i
            for item in sublist
            for i, s_item in enumerate(superlist)
            if s_item.startswith(item)
        ]

    for _og_feature in list(data.columns.values):
        # get indices of transformed features resulted from original features
        features_idx = _get_indices(sublist=[_og_feature], superlist=feature_names)
        og_feature_to_transformed_feature_names[_og_feature] = [
            feature_names[_feature_idx] for _feature_idx in features_idx
        ]

    for feature_cat_, feature_names_ in FEATURE_CATEGORY_TO_FEATURE_NAME_MAP.items():
        feature_cat_name_xai: Dict[str, float] = {}
        for feature_name_ in feature_names_:
            tf_feature_names: List[str] = og_feature_to_transformed_feature_names.get(
                feature_name_, None
            )
            if tf_feature_names:
                feature_cat_name_xai.update(
                    {
                        FEATURE_NAME_TO_TEXT_MAP[tf_feature_name_]: xai_top_k[
                            tf_feature_name_
                        ]
                        for tf_feature_name_ in tf_feature_names
                    }
                )

        # normalize values in range of (-1, 1) for each category
        total_xai_category: float = np.sum(np.abs(list(feature_cat_name_xai.values())))
        feature_cat_name_xai = {
            k: (v / total_xai_category) for k, v in feature_cat_name_xai.items()
        }

        # sort by absolute value in descending order
        feature_cat_name_xai = dict(
            sorted(
                feature_cat_name_xai.items(),
                key=lambda item: np.abs(item[1]),
                reverse=True,
            )
        )

        grouped_xai_expanded[feature_cat_.name] = feature_cat_name_xai
        # A: feature_names_
        # B: xai_top_k[feature_name_]
        # C: feature_cat_name_xai
    # X: grouped_xai_expanded

    # add manual params' responses and their importance to the result
    for param in param_composer.params:
        grouped_xai_expanded[param.feature_category.name].update(
            param.get_pprint_response_importance_dict()
        )

    return {"grouped_xai_expanded": grouped_xai_expanded}


@app.post("/grouped_xai", response_model=api_models.XaiAggregatedGroupResponse)
async def grouped_xai(features: api_models.Payload):
    # TODO: Some caching can be done here:
    # 1) `sample` is shared between all `xai` methods
    # 2) `FEATURE_CATEGORY_TO_FEATURE_NAME_MAP` can be indexed (see
    #    `aggregate_shap_values` method)

    features_dict: Dict[str, Any] = features.model_dump()

    # set response for manual params `invitation_letter`, `travel_history`, and `bank_balance`
    param_composer.set_responses_for_params(
        responses={
            invitation_letter_param.name: features.invitation_letter,
            travel_history_param.name: features.travel_history,
            bank_balance_continuous_param.name: features.bank_balance,
        },
        raw=True,
        pop=True,
        pop_containers=[features_dict],
    )

    # validate sample
    sample = _xai(**features_dict)

    # compute aggregated SHAP values for the sample
    aggregated_shap_values = flaml_tree_explainer.aggregate_shap_values(
        sample=sample,
        feature_category_to_feature_name=FEATURE_CATEGORY_TO_FEATURE_NAME_MAP,
    )

    # replace categories Enum items with their names
    for key in list(aggregated_shap_values.keys()):
        aggregated_shap_values[key.name] = aggregated_shap_values.pop(key)

    # convert shap values into normalized values in (-1, 1)
    total_xai: float = np.sum(np.abs(list(aggregated_shap_values.values())))
    for k, v in aggregated_shap_values.items():
        aggregated_shap_values[k] = v / total_xai

    # apply manual params modification given the response
    aggregated_shap_values: Dict[str, float] = param_composer.grouped_xai_modifiers(
        grouped_xai=aggregated_shap_values
    )

    return {
        "aggregated_shap_values": aggregated_shap_values,
    }


@app.post("/artificial_records")
async def generate_records(acceptance_rate: float, n: int = 5):
    record = RecordGenerator(acceptance_rate, n)
    return record.record_generator(n)


@app.get(path="/const/states", response_model=api_models.ConstantStatesResponse)
async def get_constant_states():
    """Returns all constants used in all APIs this service provides

    List of the constants this endpoint returns:

        - ``'xai_feature_categories_types'``: Returns a list of names of XAI categories.
            For example :dict:`vizard.data.constant.FEATURE_CATEGORY_TO_FEATURE_NAME_MAP`
            contains the feature names for each category.
        - ``'marriage_status_types'``: Returns a list of names of marital statuses in Canada.
            See :class:`vizard.data.constant.CanadaMarriageStatus` for more info
            for possible values.
        - ``'education_field_of_study_types'``: Returns a list of names of education field of
            study types. See :class:`vizard.data.constant.EducationFieldOfStudy` for more info
            for possible values.
        - ``'occupation_title_types'``: Returns a list of names of education field of study types.
            See :class:`vizard.data.constant.OccupationTitle` for more info for possible values.
        - ``'invitation_letter_types'``: Returns a list of names of type of relations of sender
            of invitation letter.
            See :class:`vizard.models.estimators.manual.constant.InvitationLetterSenderRelation`.
        - ``'travel_history_types'``: Returns a list of names of regions of world as travel history.
            See :class:`vizard.models.estimators.manual.constant.TravelHistoryRegion`.

    """

    return {
        "constant_states": {
            "xai_feature_categories_types": FeatureCategories.get_member_names(),
            "marriage_status_types": CanadaMarriageStatus.get_member_names(),
            "education_field_of_study_types": EducationFieldOfStudy.get_member_names(),
            "occupation_title_types": OccupationTitle.get_member_names(),
            "invitation_letter_types": list(
                InvitationLetterSenderRelation._value2member_map_.keys()
            ),
            "travel_history_types": list(TravelHistoryRegion._value2member_map_.keys()),
        }
    }


if __name__ == "__main__":
    options = {
        "bind": f"{args.bind}:{args.gunicorn_port}",
        "workers": args.workers,
        "worker_class": "uvicorn.workers.UvicornWorker",
    }
    # api_apps.StandaloneApplication(app=app, options=options).run()
    uvicorn.run(
        app=app,
        host=args.bind,
        port=args.gunicorn_port,
        log_level="debug",
        use_colors=True,
    )
