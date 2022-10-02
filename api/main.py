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
    dst_path=f'api/'
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

    return features


def _predict(*args, **kwargs):
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
            features.alias_name_indicator,
            features.sex,

            features.current_country_of_residence_country,
            features.current_country_of_residence_status,
            features.previous_country_of_residence_country2,
            features.previous_country_of_residence_country3,

            features.same_as_country_of_residence_indicator,
            features.country_where_applying_country,
            features.country_where_applying_status,

            features.previous_marriage_indicator,

            features.purpose_of_visit,
            features.funds,
            features.contact_relation_to_me,
            features.contact_relation_to_me2,

            features.education_indicator,
            features.education_field_of_study,
            features.education_country,

            features.occupation_title1,
            features.occupation_country1,
            features.occupation_title2,
            features.occupation_country2,
            features.occupation_title3,
            features.occupation_country3,

            features.no_authorized_stay,
            features.refused_entry_or_deport,
            features.previous_apply,

            features.date_of_birth,

            features.previous_country_of_residency_period2,
            features.previous_country_of_residency_period3,

            features.country_where_applying_period,  # days

            features.marriage_period,
            features.previous_marriage_period,

            features.passport_expiry_date_remaining,  # years?
            features.how_long_stay_period,  # days

            features.education_period,

            features.occupation_period,
            features.occupation_period2,
            features.occupation_period3,

            features.applicant_marital_status,
            features.mother_marital_status,
            features.father_marital_status,

            features.child_marital_status0,
            features.child_relation0,
            features.child_marital_status1,
            features.child_relation1,
            features.child_marital_status2,
            features.child_relation2,
            features.child_marital_status3,
            features.child_relation3,

            features.sibling_marital_status0,
            features.sibling_relation0,
            features.sibling_marital_status1,
            features.sibling_relation1,
            features.sibling_marital_status2,
            features.sibling_relation2,
            features.sibling_marital_status3,
            features.sibling_relation3,
            features.sibling_marital_status4,
            features.sibling_relation4,
            features.sibling_marital_status5,
            features.sibling_relation5,
            features.sibling_marital_status6,
            features.sibling_relation6,

            features.spouse_date_of_birth,
            features.mother_date_of_birth,
            features.father_date_of_birth,

            features.child_date_of_birth0,
            features.child_date_of_birth1,
            features.child_date_of_birth2,
            features.child_date_of_birth3,

            features.sibling_date_of_birth0,
            features.sibling_date_of_birth1,
            features.sibling_date_of_birth2,
            features.sibling_date_of_birth3,
            features.sibling_date_of_birth4,
            features.sibling_date_of_birth5,
            features.sibling_date_of_birth6,

            features.previous_country_of_residence_count,

            features.sibling_foreigner_count,
            features.child_mother_father_spouse_foreigner_count,

            features.child_accompany,
            features.parent_accompany,
            features.spouse_accompany,
            features.sibling_accompany,

            features.child_count,
            features.sibling_count,

            features.long_distance_child_sibling_count,
            features.foreign_living_child_sibling_count,
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
            features.alias_name_indicator,
            features.sex,

            features.current_country_of_residence_country,
            features.current_country_of_residence_status,
            features.previous_country_of_residence_country2,
            features.previous_country_of_residence_country3,

            features.same_as_country_of_residence_indicator,
            features.country_where_applying_country,
            features.country_where_applying_status,

            features.previous_marriage_indicator,

            features.purpose_of_visit,
            features.funds,
            features.contact_relation_to_me,
            features.contact_relation_to_me2,

            features.education_indicator,
            features.education_field_of_study,
            features.education_country,

            features.occupation_title1,
            features.occupation_country1,
            features.occupation_title2,
            features.occupation_country2,
            features.occupation_title3,
            features.occupation_country3,

            features.no_authorized_stay,
            features.refused_entry_or_deport,
            features.previous_apply,

            features.date_of_birth,

            features.previous_country_of_residency_period2,
            features.previous_country_of_residency_period3,

            features.country_where_applying_period,  # days

            features.marriage_period,
            features.previous_marriage_period,

            features.passport_expiry_date_remaining,  # years?
            features.how_long_stay_period,  # days

            features.education_period,

            features.occupation_period,
            features.occupation_period2,
            features.occupation_period3,

            features.applicant_marital_status,
            features.mother_marital_status,
            features.father_marital_status,

            features.child_marital_status0,
            features.child_relation0,
            features.child_marital_status1,
            features.child_relation1,
            features.child_marital_status2,
            features.child_relation2,
            features.child_marital_status3,
            features.child_relation3,

            features.sibling_marital_status0,
            features.sibling_relation0,
            features.sibling_marital_status1,
            features.sibling_relation1,
            features.sibling_marital_status2,
            features.sibling_relation2,
            features.sibling_marital_status3,
            features.sibling_relation3,
            features.sibling_marital_status4,
            features.sibling_relation4,
            features.sibling_marital_status5,
            features.sibling_relation5,
            features.sibling_marital_status6,
            features.sibling_relation6,

            features.spouse_date_of_birth,
            features.mother_date_of_birth,
            features.father_date_of_birth,

            features.child_date_of_birth0,
            features.child_date_of_birth1,
            features.child_date_of_birth2,
            features.child_date_of_birth3,

            features.sibling_date_of_birth0,
            features.sibling_date_of_birth1,
            features.sibling_date_of_birth2,
            features.sibling_date_of_birth3,
            features.sibling_date_of_birth4,
            features.sibling_date_of_birth5,
            features.sibling_date_of_birth6,

            features.previous_country_of_residence_count,

            features.sibling_foreigner_count,
            features.child_mother_father_spouse_foreigner_count,

            features.child_accompany,
            features.parent_accompany,
            features.spouse_accompany,
            features.sibling_accompany,

            features.child_count,
            features.sibling_count,

            features.long_distance_child_sibling_count,
            features.foreign_living_child_sibling_count,
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
