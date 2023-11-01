# core
import pandas as pd
import numpy as np
import pickle
from itertools import islice
# ours
from vizard.data import functional
from vizard.data import preprocessor
from vizard.data.constant import (
    CanadaContactRelation,
    CanadaResidencyStatus,
    CanadaMarriageStatus,
    EducationFieldOfStudy,
    OccupationTitle,
    FeatureCategories,
    CountryWhereApplying,
    PurposeOfVisit,
    FEATURE_CATEGORY_TO_FEATURE_NAME_MAP,
    FEATURE_NAME_TO_TEXT_MAP
)
from vizard.models import preprocessors
from vizard.models import trainers
from vizard.xai import (
    FlamlTreeExplainer,
    xai_to_text,
)
from vizard.api import apps as api_apps
from vizard.api import database as api_database
from vizard.api import models as api_models
from vizard.utils import loggers
from vizard.configs import CANADA_COUNTRY_CODE_TO_NAME
from vizard.version import VERSION as VIZARD_VERSION
# api
import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# devops
import mlflow
import dvc.api
# helpers
from typing import Dict, List, Tuple
from pathlib import Path
import argparse
import logging
import shutil
import sys


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str,
                    help='mlflow experiment name for logging',
                    default='',
                    required=True
                    )
parser.add_argument('--verbose', type=str,
                    help='logging verbosity level.',
                    choices=['debug', 'info'],
                    default='info',
                    required=True
                    )
parser.add_argument('--run_id', type=str,
                    help='MLflow run ID to extract artifacts (weights, models, etc)',
                    required=True,
                    )
parser.add_argument('--bind', type=str,
                    help='ip address of host',
                    default='0.0.0.0',
                    required=True
                    )
parser.add_argument('--mlflow_port', type=int,
                    help='port of mlflow tracking',
                    default=5000,
                    required=True
                    )
parser.add_argument('--gunicorn_port', type=int,
                    help='port used for creating gunicorn',
                    default=8000,
                    required=True
                    )
parser.add_argument('--workers', type=int,
                    help='number of works used by gunicorn',
                    default=1,
                    required=True
                    )
args = parser.parse_args()

# run mlflow tracking server
mlflow.set_tracking_uri(f'http://{args.bind}:{args.mlflow_port}')

# data versioning config
PATH = 'raw-dataset/all-dev.pkl'  # path to source data, e.g. data.pkl file
REPO = '../visaland-visa-form-utility'
VERSION = 'v2.0.1-dev'  # use the latest EDA version (i.e. `vx.x.x-dev`)
# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
data = pd.read_pickle(data_url).drop(columns=['VisaResult'], inplace=False)

# DVC: helper - (for more info see the API that uses these files)
# data file for converting country names to continuous score in "economical" sense
HELPER_PATH_GDP = 'raw-dataset/API_NY.GDP.PCAP.CD_DS2_en_xml_v2_4004943.pkl'
HELPER_VERSION_GDP = 'v0.1.0-field-GDP'  # use latest using `git tag`
# data file for converting country names to continuous score in "all" possible senses
HELPER_PATH_OVERALL = 'raw-dataset/databank-2015-2019.pkl'
HELPER_VERSION_OVERALL = 'v0.1.0-field'  # use latest using `git tag`
# gather these for MLFlow track
all_helper_data_info = {
    HELPER_PATH_GDP: HELPER_VERSION_GDP,
    HELPER_PATH_OVERALL: HELPER_VERSION_OVERALL,
}
# data file for converting country names to continuous score in "economical" sense
worldbank_gdp_dataframe = pd.read_pickle(
    dvc.api.get_url(
        path=HELPER_PATH_GDP,
        repo=REPO,
        rev=HELPER_VERSION_GDP
    )
)
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(
    dataframe=worldbank_gdp_dataframe
)
# data file for converting country names to continuous score in "all" possible senses
worldbank_overall_dataframe = pd.read_pickle(
    dvc.api.get_url(
        path=HELPER_PATH_OVERALL,
        repo=REPO,
        rev=HELPER_VERSION_OVERALL
    )
)
edu_country_score_preprocessor = preprocessor.EducationCountryScoreDataframePreprocessor(
    dataframe=worldbank_overall_dataframe
)


def residency_status_code(string: str) -> int:
    """convert residency status string to code

    Note:
        Condition on ``string`` and the resulting value derived from
        enumerator :class:`vizard.data.constant.CanadaResidencyStatus`.

    Args:
        string (str): Residency string, either ``'citizen'`` or ``'visitor'``.

    Returns:
        int: Residency code ``{1: 'citizen', 3: 'visitor', 6: 'other'}``
    """
    string = string.lower().strip()
    if string == CanadaResidencyStatus.CITIZEN.name:
        return CanadaResidencyStatus.CITIZEN.value
    elif string == CanadaResidencyStatus.VISITOR.name:
        return CanadaResidencyStatus.VISITOR.value
    elif string == CanadaResidencyStatus.OTHER.name:
        return CanadaResidencyStatus.OTHER.value
    else:
        raise ValueError(f'"{string}" is not an acceptable residency status')


# convert marital status string to code
def marital_status_code(string: str) -> int:
    string = string.lower().strip()
    if string == CanadaMarriageStatus.COMMON_LAW.name:
        return CanadaMarriageStatus.COMMON_LAW.value
    if string == CanadaMarriageStatus.DIVORCED.name:
        return CanadaMarriageStatus.DIVORCED.value
    if string == CanadaMarriageStatus.SEPARATED.name:
        return CanadaMarriageStatus.SEPARATED.value
    if string == CanadaMarriageStatus.MARRIED.name:
        return CanadaMarriageStatus.MARRIED.value
    if string == CanadaMarriageStatus.SINGLE.name:
        return CanadaMarriageStatus.SINGLE.value
    if string == CanadaMarriageStatus.WIDOWED.name:
        return CanadaMarriageStatus.WIDOWED.value
    if string == CanadaMarriageStatus.UNKNOWN.name:
        return CanadaMarriageStatus.UNKNOWN.value
    else:
        raise ValueError(f'"{string}" is not a valid marital status.')


# configure logging
VERBOSE = logging.DEBUG if args.verbose == 'debug' else logging.INFO
MLFLOW_ARTIFACTS_BASE_PATH: Path = Path('artifacts')
if MLFLOW_ARTIFACTS_BASE_PATH.exists():
    shutil.rmtree(MLFLOW_ARTIFACTS_BASE_PATH)
__libs = ['snorkel', 'vizard', 'flaml']
logger = loggers.Logger(
    name=__name__,
    level=VERBOSE,
    mlflow_artifacts_base_path=MLFLOW_ARTIFACTS_BASE_PATH,
    libs=__libs
)

# log experiment configs
if args.experiment_name == '':
    MLFLOW_EXPERIMENT_NAME = f'{VIZARD_VERSION}'
else:
    MLFLOW_EXPERIMENT_NAME = f'{args.experiment_name} - {VIZARD_VERSION}'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.start_run()

logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

# get mlflow run id for extracting artifacts of the desired run
MLFLOW_RUN_ID = args.run_id
mlflow.log_param('mlflow-trained-run-id', MLFLOW_RUN_ID)

# load fitted preprocessing models
X_CT_NAME = 'train_sklearn_column_transfer.pkl'
x_ct_path = mlflow.artifacts.download_artifacts(
    run_id=MLFLOW_RUN_ID,
    artifact_path=f'0/models/{X_CT_NAME}',
    dst_path=f'api/artifacts'
)
with open(x_ct_path, 'rb') as f:
    x_ct: preprocessors.ColumnTransformer = pickle.load(f)

# load fitted FLAML AutoML model for prediction
FLAML_AUTOML_NAME = 'flaml_automl.pkl'
flaml_automl_path = mlflow.artifacts.download_artifacts(
    run_id=MLFLOW_RUN_ID,
    artifact_path=f'0/models/{FLAML_AUTOML_NAME}',
    dst_path=f'api/artifacts'
)
with open(flaml_automl_path, 'rb') as f:
    flaml_automl: trainers.AutoML = pickle.load(f)

feature_names = preprocessors.get_transformed_feature_names(
    column_transformer=x_ct,
    original_columns_names=data.columns.values,
)

# SHAP tree explainer #56
flaml_tree_explainer = FlamlTreeExplainer(
    flaml_model=flaml_automl,
    feature_names=feature_names,
    data=None
)


# instantiate fast api app
app = fastapi.FastAPI(
    title='Vizard',
    summary='Visa chance AI assistant',
    version=VIZARD_VERSION,
    
)

# fastapi cross origin
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _preprocess(**kwargs):
    features: List = []

    # 1 P1.PD.Sex.Sex
    sex = kwargs['sex']
    features.append(sex)

    # 7 P1.PD.CWA.Row2.Country
    country_where_applying_country = kwargs['country_where_applying_country']
    features.append(country_where_applying_country)

    # 8 P1.PD.CWA.Row2.Status
    country_where_applying_status = kwargs['country_where_applying_status']
    country_where_applying_status = residency_status_code(
        country_where_applying_status
    )
    features.append(country_where_applying_status)

    # 9 P2.MS.SecA.PrevMarrIndicator
    previous_marriage_indicator = kwargs['previous_marriage_indicator']
    features.append(previous_marriage_indicator)

    # 10 P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit
    purpose_of_visit = kwargs['purpose_of_visit']
    features.append(purpose_of_visit)

    # 11 P3.DOV.PrpsRow1.Funds.Funds
    funds = kwargs['funds']
    features.append(funds)

    # 12 P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe
    contact_relation_to_me = kwargs['contact_relation_to_me']
    features.append(contact_relation_to_me)

    # 13 P3.cntcts_Row2.Relationship.RelationshipToMe
    contact_relation_to_me2 = kwargs['contact_relation_to_me2']
    features.append(contact_relation_to_me2)

    # 15 P3.Edu.Edu_Row1.FieldOfStudy
    education_field_of_study = kwargs['education_field_of_study']
    features.append(education_field_of_study)

    # 17 P3.Occ.OccRow1.Occ.Occ
    occupation_title1 = kwargs['occupation_title1']
    features.append(occupation_title1)

    # 19 P3.Occ.OccRow2.Occ.Occ
    occupation_title2 = kwargs['occupation_title2']
    features.append(occupation_title2)

    # 21 P3.Occ.OccRow3.Occ.Occ
    occupation_title3 = kwargs['occupation_title3']
    features.append(occupation_title3)

    # 23 P3.noAuthStay
    no_authorized_stay = kwargs['no_authorized_stay']
    features.append(no_authorized_stay)

    # 24 P3.refuseDeport
    refused_entry_or_deport = kwargs['refused_entry_or_deport']
    features.append(refused_entry_or_deport)

    # 25 P3.BGI2.PrevApply
    previous_apply = kwargs['previous_apply']
    features.append(previous_apply)

    # 26 P1.PD.DOBYear.Period
    date_of_birth = kwargs['date_of_birth']
    features.append(date_of_birth)

    # 29 P1.PD.CWA.Row2.Period
    country_where_applying_period = kwargs['country_where_applying_period']
    features.append(country_where_applying_period)

    # 30 P1.MS.SecA.DateOfMarr.Period
    marriage_period = kwargs['marriage_period']
    features.append(marriage_period)

    # 31 P2.MS.SecA.Period
    previous_marriage_period = kwargs['previous_marriage_period']
    features.append(previous_marriage_period)

    # 32 P2.MS.SecA.Psprt.ExpiryDate.Remaining
    passport_expiry_date_remaining = kwargs['passport_expiry_date_remaining']
    features.append(passport_expiry_date_remaining)

    # 33 P3.DOV.PrpsRow1.HLS.Period
    how_long_stay_period = kwargs['how_long_stay_period']
    features.append(how_long_stay_period)

    # 34 P3.Edu.Edu_Row1.Period
    education_period = kwargs['education_period']
    features.append(education_period)

    # 35 P3.Occ.OccRow1.Period
    occupation_period = kwargs['occupation_period']
    features.append(occupation_period)

    # 36 P3.Occ.OccRow2.Period
    occupation_period2 = kwargs['occupation_period2']
    features.append(occupation_period2)

    # 37 P3.Occ.OccRow3.Period
    occupation_period3 = kwargs['occupation_period3']
    features.append(occupation_period3)

    # 38 p1.SecA.App.ChdMStatus
    applicant_marital_status = kwargs['applicant_marital_status']
    applicant_marital_status = marital_status_code(applicant_marital_status)
    features.append(applicant_marital_status)

    # 77 VisaResult -> the label -> dropped

    # 78 P1.PD.PrevCOR.Row.Count
    previous_country_of_residence_count = kwargs['previous_country_of_residence_count']
    features.append(previous_country_of_residence_count)

    # 79 p1.SecC.Chd.X.ChdCOB.ForeignerCount
    sibling_foreigner_count = kwargs['sibling_foreigner_count']
    features.append(sibling_foreigner_count)

    # 80 p1.SecB.ChdMoFaSps.X.ChdCOB.ForeignerCount
    child_mother_father_spouse_foreigner_count = kwargs['child_mother_father_spouse_foreigner_count']
    features.append(child_mother_father_spouse_foreigner_count)

    # 81 p1.SecB.Chd.X.ChdAccomp.Count
    child_accompany = kwargs['child_accompany']
    features.append(child_accompany)

    # 82 p1.SecA.ParAccomp.Count
    parent_accompany = kwargs['parent_accompany']
    features.append(parent_accompany)

    # 83 p1.SecA.Sps.SpsAccomp.Count
    spouse_accompany = kwargs['spouse_accompany']
    features.append(spouse_accompany)

    # 84 p1.SecC.Chd.X.ChdAccomp.Count
    sibling_accompany = kwargs['sibling_accompany']
    features.append(sibling_accompany)

    # minimal-fe
    child_average_age = kwargs['child_average_age']
    features.append(child_average_age)

    # 85 p1.SecB.Chd.X.ChdRel.ChdCount
    child_count = kwargs['child_count']
    features.append(child_count)

    # minimal-fe
    sibling_average_age = kwargs['sibling_average_age']
    features.append(sibling_average_age)

    # 86 p1.SecC.Chd.X.ChdRel.ChdCount
    sibling_count = kwargs['sibling_count']
    features.append(sibling_count)

    # 87 p1.SecX.LongDistAddr
    long_distance_child_sibling_count = kwargs['long_distance_child_sibling_count']
    features.append(long_distance_child_sibling_count)

    # 88 p1.SecX.ForeignAddr
    foreign_living_child_sibling_count = kwargs['foreign_living_child_sibling_count']
    features.append(foreign_living_child_sibling_count)

    return features


def _xai(**kwargs):
    # convert api data to model data
    args = _preprocess(**kwargs)
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
        logger.debug(f'Raw data from pydantic:')
        logger.debug(f'{kwargs}\n\n')

    # convert api data to model data
    args = _preprocess(**kwargs)
    # convert to dataframe
    x_test = pd.DataFrame(data=[list(args)], columns=data.columns)
    x_test = x_test.astype(data.dtypes)
    x_test = x_test.to_numpy()
    # flag pre transformed data
    if is_flagged:
        logger.debug(f'Preprocessed but not pretransformed data:')
        logger.debug(f'{x_test}\n\n')
    
    # preprocess test data
    xt_test = x_ct.transform(x_test)
    # flag transformed data
    if is_flagged:
        logger.debug(f'Preprocessed and pretransformed data:')
        logger.debug(f'{xt_test}\n\n')
    
    # predict
    y_pred = flaml_automl.predict_proba(xt_test)
    label = np.argmax(y_pred)
    y_pred = y_pred[0, label]
    y_pred = y_pred if label == 1 else 1. - y_pred
    return y_pred


def _potential(**kwargs):
    # 1. create a one-to-one mapping from payload variables to data columns
    payload_variables: List = list(kwargs.keys())
    column_names_to_payload: Dict[str, str] = {
        column_name:payload_v for column_name, payload_v in \
            zip(list(data.columns.values), payload_variables)
    }

    # 2. create a one-to-many mapping from data columns to transformed feature names
    payload_to_transformed_feature_names: Dict[str, List[str]] = {}

    def _get_indices(
            sublist: List[str],
            superlist: List[str]
        ) -> List[int]:
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

        return [i for item in sublist for i, s_item in enumerate(superlist) if s_item.startswith(item)]


    for _og_feature in list(data.columns.values):
        # get indices of transformed features resulted from original features
        features_idx = _get_indices(
            sublist=[_og_feature],
            superlist=feature_names
        )
        payload_to_transformed_feature_names[column_names_to_payload[_og_feature]] = \
            [feature_names[_feature_idx] for _feature_idx in features_idx]

    ####### TEMP: hardcode shit

    # manipulate multiple instance variables into a single one
    payload_to_transformed_feature_names['contact_relation_to_me'] = \
        payload_to_transformed_feature_names['contact_relation_to_me'] + \
        payload_to_transformed_feature_names['contact_relation_to_me2']
    
    payload_to_transformed_feature_names['occupation_title1'] = \
        payload_to_transformed_feature_names['occupation_title1'] + \
        payload_to_transformed_feature_names['occupation_title2'] + \
        payload_to_transformed_feature_names['occupation_title3']

    payload_to_transformed_feature_names['occupation_period'] = \
        payload_to_transformed_feature_names['occupation_period'] + \
        payload_to_transformed_feature_names['occupation_period2'] + \
        payload_to_transformed_feature_names['occupation_period3']
    
    # delete merged variables
    del payload_to_transformed_feature_names['contact_relation_to_me2']
    del payload_to_transformed_feature_names['occupation_title2']
    del payload_to_transformed_feature_names['occupation_title3']
    del payload_to_transformed_feature_names['occupation_period2']
    del payload_to_transformed_feature_names['occupation_period3']

    ####### TEMP: hardcode shit

    # 3. get feature names' xai values
    xai_input: np.ndarray = _xai(**kwargs)
    # compute xai values for the sample
    xai_top_k: Dict[str, float] = flaml_tree_explainer.top_k_score(
        sample=xai_input,
        k=-1)

    # 4. provide a one-to-one mapping from payload variables to xai values
    # assign the aggregated xai value of transformed features to payload variables
    payload_to_xai: Dict[str, float] = {}
    for _payload_v, _tf_names in payload_to_transformed_feature_names.items():
        total_xai_for_tf_names: List[int] = [xai_top_k[_tf_name] for _tf_name in _tf_names]
        payload_to_xai[_payload_v] = np.sum(np.absolute(total_xai_for_tf_names)).item()
    
    return payload_to_xai


@app.post('/potential/', response_model=api_models.PotentialResponse)
async def potential(features: api_models.Payload):
    try:
        payload_to_xai = _potential(
            sex=features.sex,

            country_where_applying_country=features.country_where_applying_country,
            country_where_applying_status=features.country_where_applying_status,

            previous_marriage_indicator=features.previous_marriage_indicator,

            purpose_of_visit=features.purpose_of_visit,
            funds=features.funds,
            contact_relation_to_me=features.contact_relation_to_me,
            contact_relation_to_me2=features.contact_relation_to_me2,

            education_field_of_study=features.education_field_of_study,            

            occupation_title1=features.occupation_title1,
            occupation_title2=features.occupation_title2,            
            occupation_title3=features.occupation_title3,

            no_authorized_stay=features.no_authorized_stay,
            refused_entry_or_deport=features.refused_entry_or_deport,
            previous_apply=features.previous_apply,

            date_of_birth=features.date_of_birth,

            country_where_applying_period=features.country_where_applying_period,  # days

            marriage_period=features.marriage_period,
            previous_marriage_period=features.previous_marriage_period,

            passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
            how_long_stay_period=features.how_long_stay_period,  # days

            education_period=features.education_period,

            occupation_period=features.occupation_period,
            occupation_period2=features.occupation_period2,
            occupation_period3=features.occupation_period3,

            applicant_marital_status=features.applicant_marital_status,
            previous_country_of_residence_count=features.previous_country_of_residence_count,

            sibling_foreigner_count=features.sibling_foreigner_count,
            child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

            child_accompany=features.child_accompany,
            parent_accompany=features.parent_accompany,
            spouse_accompany=features.spouse_accompany,
            sibling_accompany=features.sibling_accompany,

            child_average_age=features.child_average_age,
            child_count=features.child_count,
            sibling_average_age=features.sibling_average_age,
            sibling_count=features.sibling_count,

            long_distance_child_sibling_count=features.long_distance_child_sibling_count,
            foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,
        )

        # calculate the potential: some of abs xai values for given variables
        # compute dictionary of payloads provided and their xai values
        provided_payload: Dict[str, float] = dict(
            (k, payload_to_xai[k]) for k in features.provided_variables if k in payload_to_xai)
        potential_by_xai_raw: float = np.sum(np.abs(list(provided_payload.values())))
        # total XAI values as the denominator (normalizer)
        total_xai: float = np.sum(np.abs(list(payload_to_xai.values())))
        # normalize to 0-1 for percentage
        potential_by_xai_normalized: float = potential_by_xai_raw / total_xai

        # TEMP: hardcoded small value to prevent 1.0 from happening just for fun
        FUN_EPSILON: float = 1e-7
        return {
            'result': potential_by_xai_normalized - FUN_EPSILON
        }

    except Exception as error:
        e = sys.exc_info()[1]
        logger.exception(e)
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@app.post('/predict/', response_model=api_models.PredictionResponse)
async def predict(
    features: api_models.Payload,
):
    try:
        result = _predict(
            sex=features.sex,

            country_where_applying_country=features.country_where_applying_country,
            country_where_applying_status=features.country_where_applying_status,

            previous_marriage_indicator=features.previous_marriage_indicator,

            purpose_of_visit=features.purpose_of_visit,
            funds=features.funds,
            contact_relation_to_me=features.contact_relation_to_me,
            contact_relation_to_me2=features.contact_relation_to_me2,

            education_field_of_study=features.education_field_of_study,            

            occupation_title1=features.occupation_title1,
            occupation_title2=features.occupation_title2,            
            occupation_title3=features.occupation_title3,

            no_authorized_stay=features.no_authorized_stay,
            refused_entry_or_deport=features.refused_entry_or_deport,
            previous_apply=features.previous_apply,

            date_of_birth=features.date_of_birth,

            country_where_applying_period=features.country_where_applying_period,  # days

            marriage_period=features.marriage_period,
            previous_marriage_period=features.previous_marriage_period,

            passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
            how_long_stay_period=features.how_long_stay_period,  # days

            education_period=features.education_period,

            occupation_period=features.occupation_period,
            occupation_period2=features.occupation_period2,
            occupation_period3=features.occupation_period3,

            applicant_marital_status=features.applicant_marital_status,
            previous_country_of_residence_count=features.previous_country_of_residence_count,

            sibling_foreigner_count=features.sibling_foreigner_count,
            child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

            child_accompany=features.child_accompany,
            parent_accompany=features.parent_accompany,
            spouse_accompany=features.spouse_accompany,
            sibling_accompany=features.sibling_accompany,

            child_average_age=features.child_average_age,
            child_count=features.child_count,
            sibling_average_age=features.sibling_average_age,
            sibling_count=features.sibling_count,

            long_distance_child_sibling_count=features.long_distance_child_sibling_count,
            foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,
        )

        # get the next question by suggesting the variable with highest XAI value
        payload_to_xai: Dict[str, float] = _potential(
            sex=features.sex,

            country_where_applying_country=features.country_where_applying_country,
            country_where_applying_status=features.country_where_applying_status,

            previous_marriage_indicator=features.previous_marriage_indicator,

            purpose_of_visit=features.purpose_of_visit,
            funds=features.funds,
            contact_relation_to_me=features.contact_relation_to_me,
            contact_relation_to_me2=features.contact_relation_to_me2,

            education_field_of_study=features.education_field_of_study,            

            occupation_title1=features.occupation_title1,
            occupation_title2=features.occupation_title2,            
            occupation_title3=features.occupation_title3,

            no_authorized_stay=features.no_authorized_stay,
            refused_entry_or_deport=features.refused_entry_or_deport,
            previous_apply=features.previous_apply,

            date_of_birth=features.date_of_birth,

            country_where_applying_period=features.country_where_applying_period,  # days

            marriage_period=features.marriage_period,
            previous_marriage_period=features.previous_marriage_period,

            passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
            how_long_stay_period=features.how_long_stay_period,  # days

            education_period=features.education_period,

            occupation_period=features.occupation_period,
            occupation_period2=features.occupation_period2,
            occupation_period3=features.occupation_period3,

            applicant_marital_status=features.applicant_marital_status,
            previous_country_of_residence_count=features.previous_country_of_residence_count,

            sibling_foreigner_count=features.sibling_foreigner_count,
            child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

            child_accompany=features.child_accompany,
            parent_accompany=features.parent_accompany,
            spouse_accompany=features.spouse_accompany,
            sibling_accompany=features.sibling_accompany,

            child_average_age=features.child_average_age,
            child_count=features.child_count,
            sibling_average_age=features.sibling_average_age,
            sibling_count=features.sibling_count,

            long_distance_child_sibling_count=features.long_distance_child_sibling_count,
            foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,
        )
        # remove variables that are in the payload (already answered)
        for provided_variable_ in features.provided_variables:
            del payload_to_xai[provided_variable_]

        next_variable: str = ''
        if payload_to_xai:
            next_variable = max(
                payload_to_xai,
                key=lambda xai_value: np.abs(payload_to_xai[xai_value])
            )
        
        logger.info('Inference finished')
        return {
            'result': result,
            'next_variable': next_variable
        }
    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@app.post('/flag/', response_model=api_models.PredictionResponse)
async def flag(
    features: api_models.Payload,
):
    is_flagged: bool = True
    # create new instance of mlflow artifact
    logger.create_artifact_instance()

    try:
        result = _predict(
            sex=features.sex,

            country_where_applying_country=features.country_where_applying_country,
            country_where_applying_status=features.country_where_applying_status,

            previous_marriage_indicator=features.previous_marriage_indicator,

            purpose_of_visit=features.purpose_of_visit,
            funds=features.funds,
            contact_relation_to_me=features.contact_relation_to_me,
            contact_relation_to_me2=features.contact_relation_to_me2,

            education_field_of_study=features.education_field_of_study,            

            occupation_title1=features.occupation_title1,
            occupation_title2=features.occupation_title2,            
            occupation_title3=features.occupation_title3,

            no_authorized_stay=features.no_authorized_stay,
            refused_entry_or_deport=features.refused_entry_or_deport,
            previous_apply=features.previous_apply,

            date_of_birth=features.date_of_birth,

            country_where_applying_period=features.country_where_applying_period,  # days

            marriage_period=features.marriage_period,
            previous_marriage_period=features.previous_marriage_period,

            passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
            how_long_stay_period=features.how_long_stay_period,  # days

            education_period=features.education_period,

            occupation_period=features.occupation_period,
            occupation_period2=features.occupation_period2,
            occupation_period3=features.occupation_period3,

            applicant_marital_status=features.applicant_marital_status,
            previous_country_of_residence_count=features.previous_country_of_residence_count,

            sibling_foreigner_count=features.sibling_foreigner_count,
            child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

            child_accompany=features.child_accompany,
            parent_accompany=features.parent_accompany,
            spouse_accompany=features.spouse_accompany,
            sibling_accompany=features.sibling_accompany,

            child_average_age=features.child_average_age,
            child_count=features.child_count,
            sibling_average_age=features.sibling_average_age,
            sibling_count=features.sibling_count,

            long_distance_child_sibling_count=features.long_distance_child_sibling_count,
            foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,

            is_flagged=is_flagged
        )

        # if need to be flagged, save as artifact
        if is_flagged:
            logger.debug(f'Features Pydantic type passed to the main endpoints:')
            logger.debug(f'{features}\n\n')
            logger.debug(f'Features dict type passed to the main endpoints:')
            logger.debug(f'{features.__dict__}\n\n')
            logger.info(f'artifacts saved in MLflow artifacts directory.')
            mlflow.log_artifacts(MLFLOW_ARTIFACTS_BASE_PATH)

        return {
            'result': result,
        }
    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@app.post('/xai', response_model=api_models.XaiResponse)
async def xai(features: api_models.Payload, k: int = 5):
    # validate sample
    sample = _xai(
        sex=features.sex,

        country_where_applying_country=features.country_where_applying_country,
        country_where_applying_status=features.country_where_applying_status,

        previous_marriage_indicator=features.previous_marriage_indicator,

        purpose_of_visit=features.purpose_of_visit,
        funds=features.funds,
        contact_relation_to_me=features.contact_relation_to_me,
        contact_relation_to_me2=features.contact_relation_to_me2,

        education_field_of_study=features.education_field_of_study,

        occupation_title1=features.occupation_title1,
        occupation_title2=features.occupation_title2,
        occupation_title3=features.occupation_title3,

        no_authorized_stay=features.no_authorized_stay,
        refused_entry_or_deport=features.refused_entry_or_deport,
        previous_apply=features.previous_apply,

        date_of_birth=features.date_of_birth,

        country_where_applying_period=features.country_where_applying_period,  # days

        marriage_period=features.marriage_period,
        previous_marriage_period=features.previous_marriage_period,

        passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
        how_long_stay_period=features.how_long_stay_period,  # days

        education_period=features.education_period,

        occupation_period=features.occupation_period,
        occupation_period2=features.occupation_period2,
        occupation_period3=features.occupation_period3,

        applicant_marital_status=features.applicant_marital_status,

        previous_country_of_residence_count=features.previous_country_of_residence_count,

        sibling_foreigner_count=features.sibling_foreigner_count,
        child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

        child_accompany=features.child_accompany,
        parent_accompany=features.parent_accompany,
        spouse_accompany=features.spouse_accompany,
        sibling_accompany=features.sibling_accompany,

        child_average_age=features.child_average_age,
        child_count=features.child_count,
        sibling_average_age=features.sibling_average_age,
        sibling_count=features.sibling_count,

        long_distance_child_sibling_count=features.long_distance_child_sibling_count,
        foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,
    )

    # compute xai values for the sample
    xai_overall_score: float = flaml_tree_explainer.overall_score(sample=sample)
    xai_top_k: Dict[str, float] = flaml_tree_explainer.top_k_score(sample=sample, k=k)

    # TODO: cannot retrieve value for transformed (let's say categorical)
    # for i, (k, v) in enumerate(xai_top_k.items()):
        # print(f'idx={i} => feat={k}, val={sample[0, i]}, xai={v}\n')

    # dict of {feature_name, xai value, textual description}
    xai_txt_top_k: Dict[str, Tuple[float, str]] = xai_to_text(
        xai_feature_values=xai_top_k,
        feature_to_keyword_mapping=FEATURE_NAME_TO_TEXT_MAP
    )

    return {
        'xai_overall_score': xai_overall_score,
        'xai_top_k': xai_top_k,
        'xai_txt_top_k': xai_txt_top_k
    }


@app.post('/grouped_xai_expanded', response_model=api_models.XaiExpandedGroupResponse)
async def xai(features: api_models.Payload):
    # validate sample
    sample = _xai(
        sex=features.sex,

        country_where_applying_country=features.country_where_applying_country,
        country_where_applying_status=features.country_where_applying_status,

        previous_marriage_indicator=features.previous_marriage_indicator,

        purpose_of_visit=features.purpose_of_visit,
        funds=features.funds,
        contact_relation_to_me=features.contact_relation_to_me,
        contact_relation_to_me2=features.contact_relation_to_me2,

        education_field_of_study=features.education_field_of_study,

        occupation_title1=features.occupation_title1,
        occupation_title2=features.occupation_title2,
        occupation_title3=features.occupation_title3,

        no_authorized_stay=features.no_authorized_stay,
        refused_entry_or_deport=features.refused_entry_or_deport,
        previous_apply=features.previous_apply,

        date_of_birth=features.date_of_birth,

        country_where_applying_period=features.country_where_applying_period,  # days

        marriage_period=features.marriage_period,
        previous_marriage_period=features.previous_marriage_period,

        passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
        how_long_stay_period=features.how_long_stay_period,  # days

        education_period=features.education_period,

        occupation_period=features.occupation_period,
        occupation_period2=features.occupation_period2,
        occupation_period3=features.occupation_period3,

        applicant_marital_status=features.applicant_marital_status,

        previous_country_of_residence_count=features.previous_country_of_residence_count,

        sibling_foreigner_count=features.sibling_foreigner_count,
        child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

        child_accompany=features.child_accompany,
        parent_accompany=features.parent_accompany,
        spouse_accompany=features.spouse_accompany,
        sibling_accompany=features.sibling_accompany,

        child_average_age=features.child_average_age,
        child_count=features.child_count,
        sibling_average_age=features.sibling_average_age,
        sibling_count=features.sibling_count,

        long_distance_child_sibling_count=features.long_distance_child_sibling_count,
        foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,
    )

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

    def _get_indices(
            sublist: List[str],
            superlist: List[str]
        ) -> List[int]:
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

        return [i for item in sublist for i, s_item in enumerate(superlist) if s_item.startswith(item)]

    for _og_feature in list(data.columns.values):
        # get indices of transformed features resulted from original features
        features_idx = _get_indices(
            sublist=[_og_feature],
            superlist=feature_names
        )
        og_feature_to_transformed_feature_names[_og_feature] = \
            [feature_names[_feature_idx] for _feature_idx in features_idx]
    
    for feature_cat_, feature_names_ in FEATURE_CATEGORY_TO_FEATURE_NAME_MAP.items():
        feature_cat_name_xai: Dict[str, float] = {}
        for feature_name_ in feature_names_:
            tf_feature_names: List[str] = og_feature_to_transformed_feature_names.get(
                feature_name_,
                None)
            if tf_feature_names:
                feature_cat_name_xai.update(
                    {
                    tf_feature_name_:xai_top_k[tf_feature_name_] \
                        for tf_feature_name_ in tf_feature_names
                    }
                )

        # normalize values in range of (-1, 1) for each category
        total_xai_category: float = np.sum(np.abs(list(feature_cat_name_xai.values())))
        feature_cat_name_xai = {
            k: (v / total_xai_category) for k, v in feature_cat_name_xai.items()
        }

        grouped_xai_expanded[feature_cat_.name] = feature_cat_name_xai
        # A: feature_names_
        # B: xai_top_k[feature_name_]
        # C: feature_cat_name_xai
    # X: grouped_xai_expanded

    return {
        'grouped_xai_expanded': grouped_xai_expanded
    }

@app.post('/grouped_xai', response_model=api_models.XaiAggregatedGroupResponse)
async def grouped_xai(features: api_models.Payload):
    # TODO: Some caching can be done here:
    # 1) `sample` is shared between all `xai` methods
    # 2) `FEATURE_CATEGORY_TO_FEATURE_NAME_MAP` can be indexed (see
    #    `aggregate_shap_values` method)

    # validate sample
    sample = _xai(
        sex=features.sex,

        country_where_applying_country=features.country_where_applying_country,
        country_where_applying_status=features.country_where_applying_status,

        previous_marriage_indicator=features.previous_marriage_indicator,

        purpose_of_visit=features.purpose_of_visit,
        funds=features.funds,
        contact_relation_to_me=features.contact_relation_to_me,
        contact_relation_to_me2=features.contact_relation_to_me2,

        education_field_of_study=features.education_field_of_study,

        occupation_title1=features.occupation_title1,
        occupation_title2=features.occupation_title2,
        occupation_title3=features.occupation_title3,

        no_authorized_stay=features.no_authorized_stay,
        refused_entry_or_deport=features.refused_entry_or_deport,
        previous_apply=features.previous_apply,

        date_of_birth=features.date_of_birth,

        country_where_applying_period=features.country_where_applying_period,  # days

        marriage_period=features.marriage_period,
        previous_marriage_period=features.previous_marriage_period,

        passport_expiry_date_remaining=features.passport_expiry_date_remaining,  # years
        how_long_stay_period=features.how_long_stay_period,  # days

        education_period=features.education_period,

        occupation_period=features.occupation_period,
        occupation_period2=features.occupation_period2,
        occupation_period3=features.occupation_period3,

        applicant_marital_status=features.applicant_marital_status,

        previous_country_of_residence_count=features.previous_country_of_residence_count,

        sibling_foreigner_count=features.sibling_foreigner_count,
        child_mother_father_spouse_foreigner_count=features.child_mother_father_spouse_foreigner_count,

        child_accompany=features.child_accompany,
        parent_accompany=features.parent_accompany,
        spouse_accompany=features.spouse_accompany,
        sibling_accompany=features.sibling_accompany,

        child_average_age=features.child_average_age,
        child_count=features.child_count,
        sibling_average_age=features.sibling_average_age,
        sibling_count=features.sibling_count,

        long_distance_child_sibling_count=features.long_distance_child_sibling_count,
        foreign_living_child_sibling_count=features.foreign_living_child_sibling_count,
    )

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

    return {
        'aggregated_shap_values': aggregated_shap_values,
    }


@app.get(
        '/const/xai_feature_categories_types',
        response_model=api_models.XaiFeatureCategoriesResponse)
async def get_xai_feature_categories_types():
    """Returns a list of names of XAI categories

    Note:
        See :class:`vizard.api.models.XaiFeatureCategoriesResponse` for more info.
    """
    return {
        'xai_feature_categories_types': FeatureCategories.get_member_names()
    }


@app.get(
    '/const/canada_marriage_status',
    response_model=api_models.CanadaMarriageStatusResponse)
async def get_canada_marriage_status():
    """Returns a list of names of marital statuses in Canada

    Note:
        See :class:`vizard.api.models.CanadaMarriageStatusResponse` for more info.
    """

    return {
        'marriage_status_types': CanadaMarriageStatus.get_member_names()
    }



@app.get(
    '/const/canada_contact_relation_types',
    response_model=api_models.CanadaContactRelationResponse)
async def get_canada_contact_relation_types():
    """Returns a list of names of contact relation types in Canada (dataset wise)

    Note:
        See :class:`vizard.api.models.CanadaContactRelationResponse` for more info.
    """

    return {
        'canada_contact_relation_types': CanadaContactRelation.get_member_names()
    }


@app.get(
    '/const/canada_residency_status_types',
    response_model=api_models.CanadaResidencyStatusResponse)
async def get_canada_residency_status_types():
    """Returns a list of names of residency status types in Canada (dataset wise)

    Note:
        See :class:`vizard.api.models.CanadaResidencyStatusResponse` for more info.
    """

    return {
        'canada_residency_status_types': CanadaResidencyStatus.get_member_names()
    }


@app.get(
    '/const/education_field_of_study_types',
    response_model=api_models.EducationFieldOfStudyResponse)
async def get_education_field_of_study_types():
    """Returns a list of names of education field of study types

    Note:
        See :class:`vizard.api.models.EducationFieldOfStudyResponse`
    """

    return {
        'education_field_of_study_types': EducationFieldOfStudy.get_member_names()
    }


@app.get(
    '/const/occupation_title_types',
    response_model=api_models.OccupationTitleResponse)
async def get_occupation_title_types():
    """Returns a list of names of education field of study types

    Note:
        See :class:`vizard.api.models.OccupationTitleResponse`
    """

    return {
        'occupation_title_types': OccupationTitle.get_member_names()
    }


@app.get(
    path='/const/country_where_applying_names',
    response_model=api_models.CountryWhereApplyingResponse)
async def get_country_where_applying_names():
    """Returns a list of names of countries user can apply from

    Note:
        See :class:`vizard.api.models.CountryWhereApplyingResponse`
    """

    return {
        'country_where_applying_names': CountryWhereApplying.get_member_names()
    }


@app.get(
    path='/const/purpose_of_visit_types',
    response_model=api_models.PurposeOfVisitResponse)
async def get_country_where_applying_names():
    """Returns a list of names of types of purposes of visit

    Note:
        See :class:`vizard.api.models.PurposeOfVisitResponse`
    """

    return {
        'purpose_of_visit_types': PurposeOfVisit.get_member_names()
    }


if __name__ == '__main__':
    options = {
        'bind': f'{args.bind}:{args.gunicorn_port}',
        'workers': args.workers,
        'worker_class': 'uvicorn.workers.UvicornWorker'
    }
    # api_apps.StandaloneApplication(app=app, options=options).run()
    uvicorn.run(
        app=app,
        host=args.bind,
        port=args.gunicorn_port,
        log_level='debug',
        use_colors=True
    )
