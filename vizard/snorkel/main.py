# core
import pandas as pd
from snorkel.augmentation.policy.core import ApplyAllPolicy
import torch
# snorkel
from snorkel.analysis import Scorer
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.augmentation import PandasTFApplier
from snorkel.augmentation import RandomPolicy
from snorkel.augmentation import preview_tfs
from snorkel.slicing import PandasSFApplier
from snorkel.slicing import slice_dataframe
# ours: snorkel
from vizard.snorkel import augmentation
from vizard.snorkel import labeling
from vizard.snorkel import modeling
from vizard.snorkel import slicing
# devops
import mlflow
import dvc.api
# helpers
import logging
import shutil
import uuid
import os


# globals
SEED = 322

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# set snorkel logger to log to our logging config
snorkel_logger = logging.getLogger('snorkel')  # simply top-level module name
snorkel_logger.setLevel(logging.INFO)

# Set up root logger, and add a file handler to root logger
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')
    os.makedirs('artifacts/logs')

log_file_name = uuid.uuid4()
logger_handler = logging.FileHandler(filename='artifacts/logs/{}-snorkel.log'.format(log_file_name),
                                     mode='w')
logger.parent.addHandler(logger_handler)  # type: ignore

logger.info(
    '\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')

# MLFlow configs
# data versioning config
PATH = 'raw-dataset/all-dev.pkl'
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.2.2-dev'
# log experiment configs
MLFLOW_EXPERIMENT_NAME = 'make modular labeling functions similar to `vizard.snorkel.augmentation`'
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
logger.info('DVC data URL used to load saved file: \n{}'.format(data_url))
logger.info('\t\t↑↑↑ Finishing reading data from DVC remote storage ↑↑↑')

logger.info('\t\t↓↓↓ Starting preparing train and test data based on labels ↓↓↓')
logger.info(
    'create *labeled* and *unlabeled* dataset and use labeled as a way to evaluate `LabelFunction`s')
output_name = 'VisaResult'
# exclude labeled data (only include weak labels and no labels)
data_unlabeled = data[(data[output_name] != 'acc') &
                      (data[output_name] != 'rej')].copy()
data_labeled = data[(data[output_name] == 'acc') |
                    (data[output_name] == 'rej')].copy()
logger.info('convert strong to weak temporary so `lf_weak_*` so `LabelFunction`s can work i.e. convert `acc` and `rej` in *labeled* dataset to `w-acc` and `w-rej`')
logger.info(
    '*remark*: the preprocessing on this data is used only for evaluation of snorkel `LabelModel`')
data_labeled[output_name] = data_labeled[output_name].apply(
    lambda x: 'w-acc' if x == 'acc' else 'w-rej')
logger.info('\t\t↑↑↑ Finishing preparing train and test data based on labels ↑↑↑')

logger.info(
    '\t\t↓↓↓ Starting extracting label matrices (L) by applying `LabelFunction`s ↓↓↓')
# labeling functions
lf_compose = [
    labeling.WeakAccept(),
    labeling.WeakReject(),
    labeling.NoIdea(),
]
lfs = labeling.ComposeLFLabeling(labelers=lf_compose)()
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
logger.info(LFAnalysis(L=label_matrix_train, lfs=lfs).lf_summary())
logger.info(
    '\t\t↑↑↑ Finishing extracting label matrices (L) by applying `LabelFunction`s ↑↑↑')

logger.info('\t\t↓↓↓ Starting training `LabelModel` ↓↓↓')
# train the label model and compute the training labels
LM_N_EPOCHS = 1000  # LM = LabelModel
LM_LOG_FREQ = 100
LM_LR = 1e-4
LM_OPTIM = 'adam'
LM_DEVICE = 'cuda'
logger.info(f'Training using device="{LM_DEVICE}"')
label_model = LabelModel(cardinality=2, verbose=True, device=LM_DEVICE)
label_model.train()
label_model.fit(label_matrix_train, n_epochs=LM_N_EPOCHS, log_freq=LM_LOG_FREQ, lr=LM_LR,
                optimizer=LM_OPTIM, seed=SEED)
logger.info('\t\t↑↑↑ Finishing training LabelModel ↑↑↑')

logger.info('\t\t↓↓↓ Starting inference on LabelModel ↓↓↓')
# test the label model
with torch.inference_mode():
    # predict labels for unlabeled data
    label_model.eval()
    auto_label_column_name = 'AL'
    logger.info('ModelLabel prediction is saved in "{}" column.'.format(
        auto_label_column_name))
    data_unlabeled.loc[:, auto_label_column_name] = label_model.predict(
        L=label_matrix_train, tie_break_policy='abstain')

    # report train accuracy (train data here is our unlabeled data)
    metrics = ['accuracy', 'coverage', 'precision', 'recall', 'f1']
    modeling.report_label_model(label_model=label_model, label_matrix=label_matrix_train,
                                gold_labels=y_train, metrics=metrics, set='train')

    # report test accuracy (test data here is our labeled data which is larger (good!))
    modeling.report_label_model(label_model=label_model, label_matrix=label_matrix_test,
                                gold_labels=y_test, metrics=metrics, set='test')
logger.info('\t\t↑↑↑ Finishing inference on LabelModel ↑↑↑')

logger.info(
    '\t\t↓↓↓ Starting augmentation by applying `TransformationFunction`s (TFs) ↓↓↓')
# transformation functions
tf_compose = [
    augmentation.AddOrderedNoiseChdAccomp(dataframe=data, sec='B'),
    augmentation.AddOrderedNoiseChdAccomp(dataframe=data, sec='C')
]
tfs = augmentation.ComposeTFAugmentation(augments=tf_compose)()  # type: ignore
# define policy for applying TFs
# TODO: #20
all_policy = ApplyAllPolicy(n_tfs=len(tfs), #sequence_length=len(tfs),
                            n_per_original=2, keep_original=True)

# apply TFs to all data (labels are not used, so no worries currently)
tf_applier = PandasTFApplier(tfs, all_policy)
data_augmented = tf_applier.apply(data)
# TF reports
logger.info(f'Original dataset size: {len(data)}')
logger.info(f'Augmented dataset size: {len(data_augmented)}')
cond1 = (data['p1.SecB.Chd.X.ChdAccomp.Count'] > 0) & (data['p1.SecB.Chd.X.ChdRel.ChdCount'] > data['p1.SecB.Chd.X.ChdAccomp.Count'])
cond2 = (data['p1.SecC.Chd.X.ChdAccomp.Count'] > 0) & (data['p1.SecC.Chd.X.ChdRel.ChdCount'] > data['p1.SecC.Chd.X.ChdAccomp.Count'])
cond = cond1 | cond2
logger.info(preview_tfs(dataframe=data[cond], tfs=tfs, n_samples=5))
logger.info('\t\t↑↑↑ Finishing augmentation by applying `TransformationFunction`s ↑↑↑')

logger.info('\t\t↓↓↓ Starting slicing by applying `SlicingFunction`s (SFs) ↓↓↓')
single_person_slice = slice_dataframe(data_augmented, slicing.single_person)
logger.info(single_person_slice.sample(5))
sfs = [slicing.single_person]
sf_applier = PandasSFApplier(sfs)
data_augmented_sliced = sf_applier.apply(data_augmented)
scorer = Scorer(metrics=metrics)
# TODO: use slicing `scorer` only for `test` set
# logger.info(scorer.score_slices(S=S_test, golds=Y_test,
#             preds=preds_test, probs=probs_test, as_dataframe=True))
logger.info('\t\t↑↑↑ Finishing slicing by applying `SlicingFunction`s (SFs) ↑↑↑')

# log data params
logger.info('\t\t↓↓↓ Starting logging hyperparams and params with MLFlow ↓↓↓')
# DVC params
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', VERSION)
mlflow.log_param('labeled_dataframe_shape', data_labeled.shape)
mlflow.log_param('unlabeled_dataframe_shape', data_unlabeled.shape)
# LabelModel params
mlflow.log_param('LabelModel_n_epochs', LM_N_EPOCHS)
mlflow.log_param('LabelModel_log_freq', LM_LOG_FREQ)
mlflow.log_param('LabelModel_lr', LM_LR)
mlflow.log_param('LabelModel_optim', LM_OPTIM)
mlflow.log_param('LabelModel_device', LM_DEVICE)
logger.info('\t\t↑↑↑ Finished logging hyperparams and params with MLFlow ↑↑↑')

# Log artifacts (logs, saved files, etc)
mlflow.log_artifacts('artifacts/')
# delete redundant logs, files that are logged as artifact
shutil.rmtree('artifacts')
