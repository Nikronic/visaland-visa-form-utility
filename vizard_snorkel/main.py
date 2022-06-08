from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

import mlflow
import dvc.api

import torch
import pandas as pd

# our packages
from vizard_snorkel import labeling
from vizard_snorkel import modeling

# utils
import logging
import os
import uuid
import shutil


# globals
SEED = 322

# configure logging
VERBOSITY = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(VERBOSITY)

logger.info(
    '\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')
# Set up root logger, and add a file handler to root logger
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')
    os.makedirs('artifacts/logs')

log_file_name = uuid.uuid4()
logger_handler = logging.FileHandler(filename='artifacts/logs/{}-snorkel.log'.format(log_file_name),
                                     mode='w')
logger.addHandler(logger_handler)

# MLFlow configs
# data versioning config
PATH = 'raw-dataset/all-dev.pkl'  # path to source data, e.g. data.pkl file
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.1.0-dev'
# log experiment configs
MLFLOW_EXPERIMENT_NAME = 'Snorkel for weak supervision of weak and unlabeled data'
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
logger.info(
    '\t\t↑↑↑ Finished setting up configs: dirs, mlflow, dvc, etc ↑↑↑')

logger.info('\t\t↓↓↓ Starting reading data from DVC remote storage ↓↓↓')
# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
# read dataset from remote (local) data storage
data = pd.read_pickle(data_url)
logger.info('\t\t↑↑↑ Finishing reading data from DVC remote storage ↑↑↑')

logger.info('\t\t↓↓↓ Starting preparing train and test data based on labels ↓↓↓')
output_name = 'VisaResult'
# exclude labeled data (only include weak labels and no labels)
data_unlabeled = data[(data[output_name] != 'acc') &
                      (data[output_name] != 'rej')]
data_labeled = data[(data[output_name] == 'acc') |
                    (data[output_name] == 'rej')]
# Remark: only should be used for evaluation of trained `LabelModel` and no where else
# convert strong to weak temporary so `lf_weak_*` can work
data_labeled[output_name] = data_labeled[output_name].apply(
    lambda x: 'w-acc' if x == 'acc' else 'w-rej')
logger.info('\t\t↑↑↑ Finishing preparing train and test data based on labels ↑↑↑')

logger.info('\t\t↓↓↓ Starting extracting label matrices (L) by applying label functions (LFs) ↓↓↓')
# label functions
lfs = [labeling.lf_weak_accept, labeling.lf_weak_reject, labeling.lf_no_idea]
# apply LFs to the unlabeled (for `LabelModel` training) and labeled (for `LabelModel` test)
applier = PandasLFApplier(lfs)
label_matrix_train = applier.apply(data_unlabeled)
# Remark: only should be used for evaluation of trained `LabelModel` and no where else
label_matrix_test = applier.apply(data_labeled)
y_test = data_labeled[output_name].apply(
    lambda x: labeling.ACC if x == 'w-acc' else labeling.REJ).values
y_train = data_unlabeled[output_name].apply(
    lambda x: labeling.ACC if x == 'w-acc' else labeling.REJ).values
# LF reports
logging.info(LFAnalysis(L=label_matrix_train, lfs=lfs).lf_summary())
logger.info('\t\t↑↑↑ Finishing extracting label matrices (L) by applying label functions (LFs) ↑↑↑')

logger.info('\t\t↓↓↓ Starting training LabelModel ↓↓↓')
# train the label model and compute the training labels
LM_N_EPOCHS = 1000  # LM = LabelModel
LM_LOG_FREQ = 100
LM_LR = 1e-4
LM_OPTIM = 'adam'
label_model = LabelModel(cardinality=2, verbose=True)
label_model.train()
label_model.fit(label_matrix_train, n_epochs=LM_N_EPOCHS, log_freq=LM_LOG_FREQ, lr=LM_LR,
                optimizer=LM_OPTIM, seed=SEED)
logger.info('\t\t↑↑↑ Finishing training LabelModel ↑↑↑')

logger.info('\t\t↓↓↓ Starting testing LabelModel ↓↓↓')
# test the label model
with torch.inference_mode():
    # predict labels for unlabeled data
    label_model.eval()
    data_unlabeled['AL'] = label_model.predict(
        L=label_matrix_train, tie_break_policy='abstain')

    # report train accuracy (train data here is our unlabeled data)
    metrics = ['accuracy', 'coverage', 'precision', 'recall', 'f1']
    modeling.report_label_model(label_model=label_model, label_matrix=label_matrix_train, 
                       gold_labels=y_train, metrics=metrics, set='train')

    # report test accuracy (test data here is our labeled data which is larger (good!))
    modeling.report_label_model(label_model=label_model, label_matrix=label_matrix_test, 
                       gold_labels=y_test, metrics=metrics, set='test')

logger.info('\t\t↑↑↑ Finishing testing LabelModel ↑↑↑')

# log data params
logger.info('\t\t↓↓↓ Starting logging with MLFlow ↓↓↓')
# DVC params
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', VERSION)
mlflow.log_param('unlabeled_dataframe_shape', data_unlabeled.shape)
# LabelModel params
mlflow.log_param('LabelModel_n_epochs', LM_N_EPOCHS)
mlflow.log_param('LabelModel_log_freq', LM_LOG_FREQ)
mlflow.log_param('LabelModel_lr', LM_LR)
mlflow.log_param('LabelModel_optim', LM_OPTIM)
logger.info('\t\t↑↑↑ Finished logging with MLFlow ↑↑↑')

# Log artifacts (logs, saved files, etc)
mlflow.log_artifacts('artifacts/')
# delete redundant logs, files that are logged as artifact
shutil.rmtree('artifacts')
