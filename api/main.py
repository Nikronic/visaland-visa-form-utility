# core
from typing import List
import pandas as pd
import numpy as np
import pickle
# ours
from vizard.data import functional
from vizard.data import preprocessor
from vizard.models import preprocessors
from vizard.models import trainers
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
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.2.5-dev'  # use the latest EDA version (i.e. `vx.x.x-dev`)
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

# instantiate fast api app
app = fastapi.FastAPI()

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
    
    # 0 P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator
    alias_name_indicator = kwargs['alias_name_indicator']
    features.append(alias_name_indicator)

    # 1 P1.PD.Sex.Sex
    sex = kwargs['sex']
    sex = sex.lower().title()  # female -> Female, ...
    features.append(sex)

    # 2 P1.PD.CurrCOR.Row2.Country
    current_country_of_residence_country = kwargs['current_country_of_residence_country']
    current_country_of_residence_country = functional.extended_dict_get(
        current_country_of_residence_country,
        functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME),
        'Unknown',
        str.isnumeric,
    )
    current_country_of_residence_country = eco_country_score_preprocessor.convert_country_name_to_numeric(
        current_country_of_residence_country
    )
    features.append(current_country_of_residence_country)

    # 3 P1.PD.CurrCOR.Row2.Status
    def f(string: str) -> int:
        """convert residency status string to code

        Args:
            string (str): Residency string, either ``'citizen'`` or ``'visitor'``.

        Returns:
            int: Residency code {1: 'citizen', 3: 'visitor'}
        """
        string = string.lower().strip()
        if string == 'citizen':
            return 1
        elif string == 'visitor':
            return 3
        elif string == 'other':
            return 6
        else:
            raise ValueError(f'"{string}" is not an acceptable residency status')
    
    current_country_of_residence_status = kwargs['current_country_of_residence_status']
    current_country_of_residence_status = f(current_country_of_residence_status)
    features.append(current_country_of_residence_status)

    # 4 P1.PD.PrevCOR.Row2.Country
    previous_country_of_residence_country2 = kwargs['previous_country_of_residence_country2']
    previous_country_of_residence_country2 = functional.extended_dict_get(
        previous_country_of_residence_country2,
        functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME),
        'Unknown',
        str.isnumeric,
    )
    previous_country_of_residence_country2 = eco_country_score_preprocessor.convert_country_name_to_numeric(
        previous_country_of_residence_country2
    )
    features.append(previous_country_of_residence_country2)

    # 5 P1.PD.PrevCOR.Row3.Country
    previous_country_of_residence_country3 = kwargs['previous_country_of_residence_country3']
    previous_country_of_residence_country3 = functional.extended_dict_get(
        previous_country_of_residence_country3,
        functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME),
        'Unknown',
        str.isnumeric,
    )
    previous_country_of_residence_country3 = eco_country_score_preprocessor.convert_country_name_to_numeric(
        previous_country_of_residence_country3
    )
    features.append(previous_country_of_residence_country3)

    # 6 P1.PD.SameAsCORIndicator
    same_as_country_of_residence_indicator = kwargs['same_as_country_of_residence_indicator']
    features.append(same_as_country_of_residence_indicator)

    # 7 P1.PD.CWA.Row2.Country
    country_where_applying_country = kwargs['country_where_applying_country']
    features.append(country_where_applying_country)

    # 8 P1.PD.CWA.Row2.Status
    country_where_applying_status = kwargs['country_where_applying_status']
    country_where_applying_status = f(country_where_applying_status)
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

    # 14 P3.Edu.EduIndicator
    education_indicator = kwargs['education_indicator']
    features.append(education_indicator)

    # 15 P3.Edu.Edu_Row1.FieldOfStudy
    education_field_of_study = kwargs['education_field_of_study']
    features.append(education_field_of_study)

    # 16 P3.Edu.Edu_Row1.Country.Country
    education_country = kwargs['education_country']
    education_country = edu_country_score_preprocessor.convert_country_name_to_numeric(
        education_country
    )
    features.append(education_country)

    # 17 P3.Occ.OccRow1.Occ.Occ
    occupation_title1 = kwargs['occupation_title1']
    # order is important, e.g. `'student of computer engineering'` should be
    #   categorized as `'student'` not `'specialist'` (because of `'eng'`)
    occ_cat_dict = {
        'manager': ['manager', 'chair', 'business', 'direct', 'owner', 'share', 'board', 'head', 'ceo'],
        'student': ['student', 'intern'],
        'retired': ['retire'],
        'specialist': ['eng', 'doc', 'med', 'arch', 'expert'],
        'employee': ['sale', 'employee', 'teacher', 'retail'],
        'housewife': ['wife'],
    }
    def categorize_occ(x: str, d: dict, default='employee') -> str:
        """if `x` found in any of `d.value`, return the corresponding `d.key`.

        Args:
            x (str): input string to search for
            d (dict): the dictionary to look for `x` in its values and return the key
            default (str, optional): if `x` no found in `d` at all. Defaults to 'employee'.
                except cases mentioned in `d` or `'OTHER'` 

        Returns:
            str: string containing a key in `d`
        """
        x = x.lower()

        # find occurrences
        for k in d.keys():
            d_vals = d[k]
            found = len([v for v in d_vals if v in x]) > 0
            if found:
                return k
        return default if x != 'other' else 'OTHER'

    occupation_title1 = categorize_occ(occupation_title1, d=occ_cat_dict, default='employee')
    features.append(occupation_title1)

    # 18 P3.Occ.OccRow1.Country.Country
    occupation_country1 = kwargs['occupation_country1']
    occupation_country1 = functional.extended_dict_get(
        occupation_country1,
        functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME),
        'Unknown',
        str.isnumeric,
    )
    occupation_country1 = eco_country_score_preprocessor.convert_country_name_to_numeric(
        occupation_country1
    )
    features.append(occupation_country1)

    # 19 P3.Occ.OccRow2.Occ.Occ
    occupation_title2 = kwargs['occupation_title2']
    occupation_title2 = categorize_occ(occupation_title2, d=occ_cat_dict, default='employee')
    features.append(occupation_title2)

    # 20 P3.Occ.OccRow2.Country.Country
    occupation_country2 = kwargs['occupation_country2']
    occupation_country2 = functional.extended_dict_get(
        occupation_country2,
        functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME),
        'Unknown',
        str.isnumeric,
    )
    occupation_country2 = eco_country_score_preprocessor.convert_country_name_to_numeric(
        occupation_country2
    )
    features.append(occupation_country2)

    # 21 P3.Occ.OccRow3.Occ.Occ
    occupation_title3 = kwargs['occupation_title3']
    occupation_title3 = categorize_occ(occupation_title3, d=occ_cat_dict, default='employee')
    features.append(occupation_title3)

    # 22 P3.Occ.OccRow3.Country.Country
    occupation_country3 = kwargs['occupation_country3']
    occupation_country3 = functional.extended_dict_get(
        occupation_country3,
        functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME),
        'Unknown',
        str.isnumeric,
    )
    occupation_country3 = eco_country_score_preprocessor.convert_country_name_to_numeric(
        occupation_country3
    )
    features.append(occupation_country3)

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
    date_of_birth = date_of_birth  # int years
    features.append(date_of_birth)

    # 27 P1.PD.PrevCOR.Row2.Period
    previous_country_of_residency_period2 = kwargs['previous_country_of_residency_period2']
    previous_country_of_residency_period2 = previous_country_of_residency_period2  # int years
    features.append(previous_country_of_residency_period2)

    # 28 P1.PD.PrevCOR.Row3.Period
    previous_country_of_residency_period3 = kwargs['previous_country_of_residency_period3']
    previous_country_of_residency_period3 = previous_country_of_residency_period3  # int years
    features.append(previous_country_of_residency_period3)

    # 29 P1.PD.CWA.Row2.Period
    country_where_applying_period = kwargs['country_where_applying_period']
    features.append(country_where_applying_period)

    # 30 P1.MS.SecA.DateOfMarr.Period
    marriage_period = kwargs['marriage_period']
    marriage_period = marriage_period  # int years
    features.append(marriage_period)

    # 31 P2.MS.SecA.Period
    previous_marriage_period = kwargs['previous_marriage_period']
    previous_marriage_period = previous_marriage_period  # int years
    features.append(previous_marriage_period)

    # 32 P2.MS.SecA.Psprt.ExpiryDate.Remaining
    passport_expiry_date_remaining = kwargs['passport_expiry_date_remaining']
    passport_expiry_date_remaining = passport_expiry_date_remaining  # int years
    features.append(passport_expiry_date_remaining)

    # 33 P3.DOV.PrpsRow1.HLS.Period
    how_long_stay_period = kwargs['how_long_stay_period']
    features.append(how_long_stay_period)

    # 34 P3.Edu.Edu_Row1.Period
    education_period = kwargs['education_period']
    education_period = education_period  # int years
    features.append(education_period)

    # 35 P3.Occ.OccRow1.Period
    occupation_period = kwargs['occupation_period']
    occupation_period = occupation_period  # int years
    features.append(occupation_period)

    # 36 P3.Occ.OccRow2.Period
    occupation_period2 = kwargs['occupation_period2']
    occupation_period2 = occupation_period2  # int years
    features.append(occupation_period2)

    # 37 P3.Occ.OccRow3.Period
    occupation_period3 = kwargs['occupation_period3']
    occupation_period3 = occupation_period3  # int years
    features.append(occupation_period3)

    # 38 p1.SecA.App.ChdMStatus
    applicant_marital_status = kwargs['applicant_marital_status']
    
    # convert marital status string to code
    def marital_status_code(string: str) -> int:
        string = string.lower().strip()
        if string == 'common-law':
            return 2
        if string == 'divorced':
            return 3
        if string == 'separated':
            return 4
        if string == 'married':
            return 5
        if string == 'single':
            return 7
        if string == 'widowed':
            return 8
        if string == 'unknown':
            return 9
        else:
            raise ValueError(f'"{string}" is not a valid marital status.') 

    applicant_marital_status = marital_status_code(applicant_marital_status)
    features.append(applicant_marital_status)

    # 39 p1.SecA.Mo.ChdMStatus
    mother_marital_status = kwargs['mother_marital_status']
    mother_marital_status = marital_status_code(mother_marital_status)
    features.append(mother_marital_status)

    # 40 p1.SecA.Fa.ChdMStatus
    father_marital_status = kwargs['father_marital_status']
    father_marital_status = marital_status_code(father_marital_status)
    features.append(father_marital_status)

    # 41 p1.SecB.Chd.[0].ChdMStatus
    child_marital_status0 = kwargs['child_marital_status0']
    child_marital_status0 = marital_status_code(child_marital_status0)
    features.append(child_marital_status0)

    # 42 p1.SecB.Chd.[0].ChdRel
    child_relation0 = kwargs['child_relation0']
    features.append(child_relation0)

    # 43 p1.SecB.Chd.[1].ChdMStatus
    child_marital_status1 = kwargs['child_marital_status1']
    child_marital_status1 = marital_status_code(child_marital_status1)
    features.append(child_marital_status1)

    # 44 p1.SecB.Chd.[1].ChdRel
    child_relation1 = kwargs['child_relation1']
    features.append(child_relation1)

    # 45 p1.SecB.Chd.[2].ChdMStatus
    child_marital_status2 = kwargs['child_marital_status2']
    child_marital_status2 = marital_status_code(child_marital_status2)
    features.append(child_marital_status2)

    # 46 p1.SecB.Chd.[2].ChdRel
    child_relation2 = kwargs['child_relation2']
    features.append(child_relation2)

    # 47 p1.SecB.Chd.[3].ChdMStatus
    child_marital_status3 = kwargs['child_marital_status3']
    child_marital_status3 = marital_status_code(child_marital_status3)
    features.append(child_marital_status3)

    # 48 p1.SecB.Chd.[3].ChdRel
    child_relation3 = kwargs['child_relation3']
    features.append(child_relation3)

    # 49 p1.SecC.Chd.[0].ChdMStatus
    sibling_marital_status0 = kwargs['sibling_marital_status0']
    sibling_marital_status0 = marital_status_code(sibling_marital_status0)
    features.append(sibling_marital_status0)

    # 50 p1.SecC.Chd.[0].ChdRel
    sibling_relation0 = kwargs['sibling_relation0']
    features.append(sibling_relation0)

    # 51 p1.SecC.Chd.[1].ChdMStatus
    sibling_marital_status1 = kwargs['sibling_marital_status1']
    sibling_marital_status1 = marital_status_code(sibling_marital_status1)
    features.append(sibling_marital_status1)

    # 52 p1.SecC.Chd.[1].ChdRel
    sibling_relation1 = kwargs['sibling_relation1']
    features.append(sibling_relation1)

    # 53 p1.SecC.Chd.[2].ChdMStatus
    sibling_marital_status2 = kwargs['sibling_marital_status2']
    sibling_marital_status2 = marital_status_code(sibling_marital_status2)
    features.append(sibling_marital_status2)

    # 54 p1.SecC.Chd.[2].ChdRel
    sibling_relation2 = kwargs['sibling_relation2']
    features.append(sibling_relation2)

    # 55 p1.SecC.Chd.[3].ChdMStatus
    sibling_marital_status3 = kwargs['sibling_marital_status3']
    sibling_marital_status3 = marital_status_code(sibling_marital_status3)
    features.append(sibling_marital_status3)

    # 56 p1.SecC.Chd.[3].ChdRel
    sibling_relation3 = kwargs['sibling_relation3']
    features.append(sibling_relation3)

    # 57 p1.SecC.Chd.[4].ChdMStatus
    sibling_marital_status4 = kwargs['sibling_marital_status4']
    sibling_marital_status4 = marital_status_code(sibling_marital_status4)
    features.append(sibling_marital_status4)

    # 58 p1.SecC.Chd.[4].ChdRel
    sibling_relation4 = kwargs['sibling_relation4']
    features.append(sibling_relation4)

    # 59 p1.SecC.Chd.[5].ChdMStatus
    sibling_marital_status5 = kwargs['sibling_marital_status5']
    sibling_marital_status5 = marital_status_code(sibling_marital_status5)
    features.append(sibling_marital_status5)

    # 60 p1.SecC.Chd.[5].ChdRel
    sibling_relation5 = kwargs['sibling_relation5']
    features.append(sibling_relation5)

    # 61 p1.SecC.Chd.[6].ChdMStatus
    sibling_marital_status6 = kwargs['sibling_marital_status6']
    sibling_marital_status6 = marital_status_code(sibling_marital_status6)
    features.append(sibling_marital_status6)

    # 62 p1.SecC.Chd.[6].ChdRel
    sibling_relation6 = kwargs['sibling_relation6']
    features.append(sibling_relation6)

    # 63 p1.SecA.Sps.SpsDOB.Period
    spouse_date_of_birth = kwargs['spouse_date_of_birth']
    spouse_date_of_birth = spouse_date_of_birth  # int years
    features.append(spouse_date_of_birth)

    # 64 p1.SecA.Mo.MoDOB.Period
    mother_date_of_birth = kwargs['mother_date_of_birth']
    mother_date_of_birth = mother_date_of_birth  # int years
    features.append(mother_date_of_birth)

    # 65 p1.SecA.Fa.FaDOB.Period
    father_date_of_birth = kwargs['father_date_of_birth']
    father_date_of_birth = father_date_of_birth  # int years
    features.append(father_date_of_birth)

    # 66 p1.SecB.Chd.[0].ChdDOB.Period
    child_date_of_birth0 = kwargs['child_date_of_birth0']
    child_date_of_birth0 = child_date_of_birth0  # int years
    features.append(child_date_of_birth0)

    # 67 p1.SecB.Chd.[1].ChdDOB.Period
    child_date_of_birth1 = kwargs['child_date_of_birth1']
    child_date_of_birth1 = child_date_of_birth1  # int years
    features.append(child_date_of_birth1)

    # 68 p1.SecB.Chd.[2].ChdDOB.Period
    child_date_of_birth2 = kwargs['child_date_of_birth2']
    child_date_of_birth2 = child_date_of_birth2  # int years
    features.append(child_date_of_birth2)

    # 69 p1.SecB.Chd.[3].ChdDOB.Period
    child_date_of_birth3 = kwargs['child_date_of_birth3']
    child_date_of_birth3 = child_date_of_birth3  # int years
    features.append(child_date_of_birth3)

    # 70 p1.SecC.Chd.[0].ChdDOB.Period 
    sibling_date_of_birth0 = kwargs['sibling_date_of_birth0']
    sibling_date_of_birth0 = sibling_date_of_birth0  # int years
    features.append(sibling_date_of_birth0)

    # 71 p1.SecC.Chd.[1].ChdDOB.Period 
    sibling_date_of_birth1 = kwargs['sibling_date_of_birth1']
    sibling_date_of_birth1 = sibling_date_of_birth1  # int years
    features.append(sibling_date_of_birth1)

    # 72 p1.SecC.Chd.[2].ChdDOB.Period 
    sibling_date_of_birth2 = kwargs['sibling_date_of_birth2']
    sibling_date_of_birth2 = sibling_date_of_birth2  # int years
    features.append(sibling_date_of_birth2)

    # 73 p1.SecC.Chd.[3].ChdDOB.Period 
    sibling_date_of_birth3 = kwargs['sibling_date_of_birth3']
    sibling_date_of_birth3 = sibling_date_of_birth3  # int years
    features.append(sibling_date_of_birth3)

    # 74 p1.SecC.Chd.[4].ChdDOB.Period 
    sibling_date_of_birth4 = kwargs['sibling_date_of_birth4']
    sibling_date_of_birth4 = sibling_date_of_birth4  # int years
    features.append(sibling_date_of_birth4)

    # 75 p1.SecC.Chd.[5].ChdDOB.Period 
    sibling_date_of_birth5 = kwargs['sibling_date_of_birth5']
    sibling_date_of_birth5 = sibling_date_of_birth5  # int years
    features.append(sibling_date_of_birth5)

    # 76 p1.SecC.Chd.[6].ChdDOB.Period 
    sibling_date_of_birth6 = kwargs['sibling_date_of_birth6']
    sibling_date_of_birth6 = sibling_date_of_birth6  # int years
    features.append(sibling_date_of_birth6)

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

    # 85 p1.SecB.Chd.X.ChdRel.ChdCount
    child_count = kwargs['child_count']
    features.append(child_count)

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


def _predict(**kwargs):
    # convert api data to model data
    args = _preprocess(**kwargs)
    # convert to dataframe
    x_test = pd.DataFrame(data=[list(args)], columns=data.columns)
    x_test = x_test.astype(data.dtypes)
    x_test = x_test.to_numpy()
    # preprocess test data
    xt_test = x_ct.transform(x_test)
    # predict
    y_pred = flaml_automl.predict_proba(xt_test)
    label = np.argmax(y_pred)
    y_pred = y_pred[0, label]
    y_pred = y_pred if label == 1 else 1. - y_pred
    return y_pred


@app.post('/predict/', response_model=api_models.PredictionResponse)
async def predict(
    features: api_models.Payload,
):
    try:
        result = _predict(
            alias_name_indicator = features.alias_name_indicator,
            sex = features.sex,

            current_country_of_residence_country = features.current_country_of_residence_country,
            current_country_of_residence_status = features.current_country_of_residence_status,
            previous_country_of_residence_country2 = features.previous_country_of_residence_country2,
            previous_country_of_residence_country3 = features.previous_country_of_residence_country3,

            same_as_country_of_residence_indicator = features.same_as_country_of_residence_indicator,
            country_where_applying_country = features.country_where_applying_country,
            country_where_applying_status = features.country_where_applying_status,

            previous_marriage_indicator = features.previous_marriage_indicator,

            purpose_of_visit = features.purpose_of_visit,
            funds = features.funds,
            contact_relation_to_me = features.contact_relation_to_me,
            contact_relation_to_me2 = features.contact_relation_to_me2,

            education_indicator = features.education_indicator,
            education_field_of_study = features.education_field_of_study,
            education_country = features.education_country,

            occupation_title1 = features.occupation_title1,
            occupation_country1 = features.occupation_country1,
            occupation_title2 = features.occupation_title2,
            occupation_country2 = features.occupation_country2,
            occupation_title3 = features.occupation_title3,
            occupation_country3 = features.occupation_country3,

            no_authorized_stay = features.no_authorized_stay,
            refused_entry_or_deport = features.refused_entry_or_deport,
            previous_apply = features.previous_apply,

            date_of_birth = features.date_of_birth,

            previous_country_of_residency_period2 = features.previous_country_of_residency_period2,
            previous_country_of_residency_period3 = features.previous_country_of_residency_period3,

            country_where_applying_period = features.country_where_applying_period,  # days

            marriage_period = features.marriage_period,
            previous_marriage_period = features.previous_marriage_period,

            passport_expiry_date_remaining = features.passport_expiry_date_remaining,  # years?
            how_long_stay_period = features.how_long_stay_period,  # days

            education_period = features.education_period,

            occupation_period = features.occupation_period,
            occupation_period2 = features.occupation_period2,
            occupation_period3 = features.occupation_period3,

            applicant_marital_status = features.applicant_marital_status,
            mother_marital_status = features.mother_marital_status,
            father_marital_status = features.father_marital_status,

            child_marital_status0 = features.child_marital_status0,
            child_relation0 = features.child_relation0,
            child_marital_status1 = features.child_marital_status1,
            child_relation1 = features.child_relation1,
            child_marital_status2 = features.child_marital_status2,
            child_relation2 = features.child_relation2,
            child_marital_status3 = features.child_marital_status3,
            child_relation3 = features.child_relation3,

            sibling_marital_status0 = features.sibling_marital_status0,
            sibling_relation0 = features.sibling_relation0,
            sibling_marital_status1 = features.sibling_marital_status1,
            sibling_relation1 = features.sibling_relation1,
            sibling_marital_status2 = features.sibling_marital_status2,
            sibling_relation2 = features.sibling_relation2,
            sibling_marital_status3 = features.sibling_marital_status3,
            sibling_relation3 = features.sibling_relation3,
            sibling_marital_status4 = features.sibling_marital_status4,
            sibling_relation4 = features.sibling_relation4,
            sibling_marital_status5 = features.sibling_marital_status5,
            sibling_relation5 = features.sibling_relation5,
            sibling_marital_status6 = features.sibling_marital_status6,
            sibling_relation6 = features.sibling_relation6,

            spouse_date_of_birth = features.spouse_date_of_birth,
            mother_date_of_birth = features.mother_date_of_birth,
            father_date_of_birth = features.father_date_of_birth,

            child_date_of_birth0 = features.child_date_of_birth0,
            child_date_of_birth1 = features.child_date_of_birth1,
            child_date_of_birth2 = features.child_date_of_birth2,
            child_date_of_birth3 = features.child_date_of_birth3,

            sibling_date_of_birth0 = features.sibling_date_of_birth0,
            sibling_date_of_birth1 = features.sibling_date_of_birth1,
            sibling_date_of_birth2 = features.sibling_date_of_birth2,
            sibling_date_of_birth3 = features.sibling_date_of_birth3,
            sibling_date_of_birth4 = features.sibling_date_of_birth4,
            sibling_date_of_birth5 = features.sibling_date_of_birth5,
            sibling_date_of_birth6 = features.sibling_date_of_birth6,

            previous_country_of_residence_count = features.previous_country_of_residence_count,

            sibling_foreigner_count = features.sibling_foreigner_count,
            child_mother_father_spouse_foreigner_count = features.child_mother_father_spouse_foreigner_count,

            child_accompany = features.child_accompany,
            parent_accompany = features.parent_accompany,
            spouse_accompany = features.spouse_accompany,
            sibling_accompany = features.sibling_accompany,

            child_count = features.child_count,
            sibling_count = features.sibling_count,

            long_distance_child_sibling_count = features.long_distance_child_sibling_count,
            foreign_living_child_sibling_count = features.foreign_living_child_sibling_count,
        )

        logger.info('Inference finished')
        return {
            'result': result,
        }
    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@app.post('/flag/', response_model=api_models.PredictionResponse)
async def flag(
    features: api_models.Payload,
):

    # create new instance of mlflow artifact
    logger.create_artifact_instance()

    try:
        result = _predict(
            alias_name_indicator = features.alias_name_indicator,
            sex = features.sex,

            current_country_of_residence_country = features.current_country_of_residence_country,
            current_country_of_residence_status = features.current_country_of_residence_status,
            previous_country_of_residence_country2 = features.previous_country_of_residence_country2,
            previous_country_of_residence_country3 = features.previous_country_of_residence_country3,

            same_as_country_of_residence_indicator = features.same_as_country_of_residence_indicator,
            country_where_applying_country = features.country_where_applying_country,
            country_where_applying_status = features.country_where_applying_status,

            previous_marriage_indicator = features.previous_marriage_indicator,

            purpose_of_visit = features.purpose_of_visit,
            funds = features.funds,
            contact_relation_to_me = features.contact_relation_to_me,
            contact_relation_to_me2 = features.contact_relation_to_me2,

            education_indicator = features.education_indicator,
            education_field_of_study = features.education_field_of_study,
            education_country = features.education_country,

            occupation_title1 = features.occupation_title1,
            occupation_country1 = features.occupation_country1,
            occupation_title2 = features.occupation_title2,
            occupation_country2 = features.occupation_country2,
            occupation_title3 = features.occupation_title3,
            occupation_country3 = features.occupation_country3,

            no_authorized_stay = features.no_authorized_stay,
            refused_entry_or_deport = features.refused_entry_or_deport,
            previous_apply = features.previous_apply,

            date_of_birth = features.date_of_birth,

            previous_country_of_residency_period2 = features.previous_country_of_residency_period2,
            previous_country_of_residency_period3 = features.previous_country_of_residency_period3,

            country_where_applying_period = features.country_where_applying_period,  # days

            marriage_period = features.marriage_period,
            previous_marriage_period = features.previous_marriage_period,

            passport_expiry_date_remaining = features.passport_expiry_date_remaining,  # years?
            how_long_stay_period = features.how_long_stay_period,  # days

            education_period = features.education_period,

            occupation_period = features.occupation_period,
            occupation_period2 = features.occupation_period2,
            occupation_period3 = features.occupation_period3,

            applicant_marital_status = features.applicant_marital_status,
            mother_marital_status = features.mother_marital_status,
            father_marital_status = features.father_marital_status,

            child_marital_status0 = features.child_marital_status0,
            child_relation0 = features.child_relation0,
            child_marital_status1 = features.child_marital_status1,
            child_relation1 = features.child_relation1,
            child_marital_status2 = features.child_marital_status2,
            child_relation2 = features.child_relation2,
            child_marital_status3 = features.child_marital_status3,
            child_relation3 = features.child_relation3,

            sibling_marital_status0 = features.sibling_marital_status0,
            sibling_relation0 = features.sibling_relation0,
            sibling_marital_status1 = features.sibling_marital_status1,
            sibling_relation1 = features.sibling_relation1,
            sibling_marital_status2 = features.sibling_marital_status2,
            sibling_relation2 = features.sibling_relation2,
            sibling_marital_status3 = features.sibling_marital_status3,
            sibling_relation3 = features.sibling_relation3,
            sibling_marital_status4 = features.sibling_marital_status4,
            sibling_relation4 = features.sibling_relation4,
            sibling_marital_status5 = features.sibling_marital_status5,
            sibling_relation5 = features.sibling_relation5,
            sibling_marital_status6 = features.sibling_marital_status6,
            sibling_relation6 = features.sibling_relation6,

            spouse_date_of_birth = features.spouse_date_of_birth,
            mother_date_of_birth = features.mother_date_of_birth,
            father_date_of_birth = features.father_date_of_birth,

            child_date_of_birth0 = features.child_date_of_birth0,
            child_date_of_birth1 = features.child_date_of_birth1,
            child_date_of_birth2 = features.child_date_of_birth2,
            child_date_of_birth3 = features.child_date_of_birth3,

            sibling_date_of_birth0 = features.sibling_date_of_birth0,
            sibling_date_of_birth1 = features.sibling_date_of_birth1,
            sibling_date_of_birth2 = features.sibling_date_of_birth2,
            sibling_date_of_birth3 = features.sibling_date_of_birth3,
            sibling_date_of_birth4 = features.sibling_date_of_birth4,
            sibling_date_of_birth5 = features.sibling_date_of_birth5,
            sibling_date_of_birth6 = features.sibling_date_of_birth6,

            previous_country_of_residence_count = features.previous_country_of_residence_count,

            sibling_foreigner_count = features.sibling_foreigner_count,
            child_mother_father_spouse_foreigner_count = features.child_mother_father_spouse_foreigner_count,

            child_accompany = features.child_accompany,
            parent_accompany = features.parent_accompany,
            spouse_accompany = features.spouse_accompany,
            sibling_accompany = features.sibling_accompany,

            child_count = features.child_count,
            sibling_count = features.sibling_count,

            long_distance_child_sibling_count = features.long_distance_child_sibling_count,
            foreign_living_child_sibling_count = features.foreign_living_child_sibling_count,
        )

        # if need to be flagged, save as artifact
        if flag:
            logger.info(f'artifacts saved in MLflow artifacts directory.')
            mlflow.log_artifacts(MLFLOW_ARTIFACTS_BASE_PATH)

        return {
            'result': result,
        }
    except Exception as error:
        logger.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))


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
        debug=True
    )
