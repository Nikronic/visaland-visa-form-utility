# core
import enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
# ours: models
from vizard.models import estimators
from vizard.models import evaluators
from vizard.models import preprocessors
from vizard.models import trainers
from vizard.models import weights
# devops
import mlflow
import dvc.api
# helpers
from pathlib import Path
import logging
import shutil
import os


# globals
SEED = 322
VERBOSE = logging.DEBUG

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(VERBOSE)
# set `models` logger to log to our logging config
models_logger = logging.getLogger('vizard.models')
models_logger.setLevel(VERBOSE)

# Set up root logger, and add a file handler to root logger
MLFLOW_ARTIFACTS_PATH = Path('artifacts')
MLFLOW_ARTIFACTS_LOGS_PATH = MLFLOW_ARTIFACTS_PATH / 'logs'
MLFLOW_ARTIFACTS_CONFIGS_PATH = MLFLOW_ARTIFACTS_PATH / 'configs'
if not os.path.exists(MLFLOW_ARTIFACTS_PATH):
    os.makedirs(MLFLOW_ARTIFACTS_PATH)
    os.makedirs(MLFLOW_ARTIFACTS_LOGS_PATH)
    os.makedirs(MLFLOW_ARTIFACTS_CONFIGS_PATH)

logger_handler = logging.FileHandler(filename=MLFLOW_ARTIFACTS_LOGS_PATH / 'models.log',
                                     mode='w')
logger.parent.addHandler(logger_handler)  # type: ignore
logger.info('\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')

# MLFlow configs
# data versioning config
PATH = 'raw-dataset/all-dev.pkl'
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.2.2-dev'
# log experiment configs
MLFLOW_EXPERIMENT_NAME = 'setup modeling pipeline - transforming Y'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_TAGS = {
    'stage': 'dev'  # dev, beta, production
}
mlflow.set_tags(MLFLOW_TAGS)

logger.info('MLflow experiment name: {}'.format(MLFLOW_EXPERIMENT_NAME))
logger.info('MLflow experiment id: {}'.format(mlflow.active_run().info.run_id))
logger.info('MLflow data version: {}'.format(VERSION))
logger.info('MLflow repo (root): {}'.format(REPO))
logger.info('MLflow data source path: {}'.format(PATH))
logger.info('\t\t↑↑↑ Finished setting up configs: dirs, mlflow, dvc, etc ↑↑↑')

logger.info('\t\t↓↓↓ Starting reading data from DVC remote storage ↓↓↓')
# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
# read dataset from remote (local) data storage
data = pd.read_pickle(data_url)
logger.info('DVC data URL used to load saved file: \n{}'.format(data_url))
logger.info('\t\t↑↑↑ Finishing reading data from DVC remote storage ↑↑↑')

logger.info(
    '\t\t↓↓↓ Starting preprocessing on directly DVC `vX.X.X-dev` data ↓↓↓')
# TODO: add preprocessing steps here

# move the dependent variable to the end of the dataframe
data = preprocessors.move_dependent_variable_to_end(
    df=data, target_column='VisaResult')

# convert to np and split to train, test, eval
train_test_eval_splitter = preprocessors.TrainTestEvalSplit(random_state=SEED)
data_tuple = train_test_eval_splitter(df=data, target_column='VisaResult')
x_train, x_test, x_eval, y_train, y_test, y_eval = data_tuple

# Transform and normalize appropriately given config
x_column_transformers_config = preprocessors.ColumnTransformerConfig()
x_column_transformers_config.set_configs(
    preprocessors.CANADA_COLUMN_TRANSFORMER_CONFIG_X)

x_ct = preprocessors.ColumnTransformer(
    transformers=x_column_transformers_config.generate_pipeline(df=data),
    remainder='passthrough',
    verbose=False,
    verbose_feature_names_out=False,
    n_jobs=None,
)

xt_train = x_ct.fit_transform(x_train)
yt_train = LabelBinarizer().fit_transform(y_train)  # TODO: see #54

# preview the transformed data
preview_ct = preprocessors.preview_column_transformer(column_transformer=x_ct,
                                                      original=x_train,
                                                      transformed=xt_train,
                                                      df=data,
                                                      random_state=SEED,
                                                      n_samples=1)
logger.info([_ for _ in preview_ct])

logger.info(
    '\t\t↑↑↑ Finished preprocessing on directly DVC `vX.X.X-dev` data ↑↑↑')

logger.info('\t\t↓↓↓ Starting defining estimators models ↓↓↓')
# TODO: add estimators definition here
logger.info('\t\t↑↑↑ Finished defining estimators models ↑↑↑')

logger.info(
    '\t\t↓↓↓ Starting loading training config and training estimators ↓↓↓')
# TODO: add training steps here
logger.info(
    '\t\t↑↑↑ Finished loading training config and training estimators ↑↑↑')

logger.info(
    '\t\t↓↓↓ Starting loading evaluation config and evaluating estimators ↓↓↓')
# TODO: add final evaluation steps here
logger.info(
    '\t\t↑↑↑ Finished loading evaluation config and evaluating estimators ↑↑↑')

logger.info('\t\t↓↓↓ Starting saving good weights ↓↓↓')
# TODO: add final checkpoint here (save weights)
logger.info('\t\t↑↑↑ Finished saving good weights ↑↑↑')

logger.info('\t\t↓↓↓ Starting logging preview of results and other stuff ↓↓↓')
# TODO: add final checkpoint here (save weights)
logger.info('\t\t↑↑↑ Finished logging preview of results and other stuff ↑↑↑')

# log data params
logger.info('\t\t↓↓↓ Starting logging hyperparams and params with MLFlow ↓↓↓')
# DVC params
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', VERSION)
# TODO: log preprocessor configs
# TODO: log estimator params
# TODO: log trainer config
# TODO: log evaluator config
# TODO: log weights
# TODO: log anything else in between that needs to be logged
# preprocessing parameters params
mlflow.log_param('x_train_shape', x_train.shape)
mlflow.log_param('xt_train_shape', xt_train.shape)
mlflow.log_param('x_test_shape', x_test.shape)
mlflow.log_param('x_val_shape', x_eval.shape)
mlflow.log_param('y_train_shape', y_train.shape)
mlflow.log_param('yt_train_shape', yt_train.shape)
mlflow.log_param('y_test_shape', y_test.shape)
mlflow.log_param('y_val_shape', y_test.shape)
logger.info('\t\t↑↑↑ Finished logging hyperparams and params with MLFlow ↑↑↑')

# dump all json configs into artifacts
x_column_transformers_config.as_mlflow_artifact(MLFLOW_ARTIFACTS_CONFIGS_PATH)
train_test_eval_splitter.as_mlflow_artifact(MLFLOW_ARTIFACTS_CONFIGS_PATH)
# Log artifacts (logs, saved files, etc)
mlflow.log_artifacts(MLFLOW_ARTIFACTS_PATH)
# delete redundant logs, files that are logged as artifact
shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
