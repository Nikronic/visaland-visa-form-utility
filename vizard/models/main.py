# core
import pandas as pd
import torch
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
import logging
import shutil
import os


# globals
SEED = 322

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# set snorkel logger to log to our logging config
snorkel_logger = logging.getLogger('models')  # simply top-level module name
snorkel_logger.setLevel(logging.INFO)

# Set up root logger, and add a file handler to root logger
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')
    os.makedirs('artifacts/logs')

logger_handler = logging.FileHandler(filename='artifacts/logs/models.log',
                                     mode='w')
logger.parent.addHandler(logger_handler)  # type: ignore
logger.info('\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')

# MLFlow configs
# data versioning config
PATH = 'raw-dataset/all-dev.pkl'
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.2.2-dev'
# log experiment configs
MLFLOW_EXPERIMENT_NAME = 'setup modeling pipeline'
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

logger.info('\t\t↓↓↓ Starting preprocessing on directly DVC `vX.X.X-dev` data ↓↓↓')
# TODO: add preprocessing steps here
logger.info('\t\t↑↑↑ Finished preprocessing on directly DVC `vX.X.X-dev` data ↑↑↑')

logger.info('\t\t↓↓↓ Starting defining estimators models ↓↓↓')
# TODO: add estimators definition here
logger.info('\t\t↑↑↑ Finished defining estimators models ↑↑↑')

logger.info('\t\t↓↓↓ Starting loading training config and training estimators ↓↓↓')
# TODO: add training steps here
logger.info('\t\t↑↑↑ Finished loading training config and training estimators ↑↑↑')

logger.info('\t\t↓↓↓ Starting loading evaluation config and evaluating estimators ↓↓↓')
# TODO: add final evaluation steps here
logger.info('\t\t↑↑↑ Finished loading evaluation config and evaluating estimators ↑↑↑')

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
# mlflow.log_param('labeled_dataframe_shape', data_labeled.shape)
# mlflow.log_param('unlabeled_dataframe_shape', data_unlabeled.shape)
# # LabelModel params
# mlflow.log_param('LabelModel_n_epochs', LM_N_EPOCHS)
# mlflow.log_param('LabelModel_log_freq', LM_LOG_FREQ)
# mlflow.log_param('LabelModel_lr', LM_LR)
# mlflow.log_param('LabelModel_optim', LM_OPTIM)
# mlflow.log_param('LabelModel_device', LM_DEVICE)
logger.info('\t\t↑↑↑ Finished logging hyperparams and params with MLFlow ↑↑↑')

# Log artifacts (logs, saved files, etc)
mlflow.log_artifacts('artifacts/')
# delete redundant logs, files that are logged as artifact
shutil.rmtree('artifacts')
