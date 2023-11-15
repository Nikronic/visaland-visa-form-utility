# core
import pandas as pd
import torch
# ours: snorkel
from vizard.snorkel import LABEL_MODEL_CONFIGS
from vizard.snorkel import labeling
from vizard.snorkel import modeling
from vizard.snorkel import augmentation
from vizard.snorkel import PandasLFApplier
from vizard.snorkel import PandasTFApplier
from vizard.snorkel import LFAnalysis
from vizard.snorkel import LabelModel
from vizard.snorkel import ApplyAllPolicy
# ours: data
from vizard.data.constant import ClassificationLabels
# ours: models
from vizard.models import preprocessors
from vizard.models import trainers
from vizard.models.trainers.aml_flaml import EvalMode
# ours: helpers
from vizard.version import VERSION as VIZARD_VERSION
# from vizard.utils.dtreeviz import FLAMLDTreeViz
from vizard.configs import JsonConfigHandler
from vizard.utils import loggers
# devops
import dvc.api
import mlflow
# helpers
from typing import Any, Tuple, List
from pathlib import Path
import enlighten
import argparse
import logging
import pickle
import shutil
import sys


# argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dvc_data_path',
    type=str,
    help='path to DVC versioned processed data source',
    default='raw-dataset/all-dev.pkl',
    required=True)
parser.add_argument(
    '--dvc_repo',
    type=str,
    help='repo associated with DVC',
    default='../visaland-visa-form-utility',
    required=True)
parser.add_argument(
    '--dvc_data_version',
    type=str,
    help='version of DVC versioned data source (i.e., version of `dvc_data_path`)',
    default='v2.0.1-dev',
    required=True)
parser.add_argument(
    '-e',
    '--experiment_name',
    type=str,
    help='mlflow experiment name for logging',
    default='',
    required=False)
parser.add_argument(
    '-d',
    '--verbose',
    type=str,
    help='logging verbosity level.',
    choices=['debug', 'info'],
    default='info',
    required=False)
parser.add_argument(
    '-b',
    '--bind',
    type=str,
    help='ip address of host',
    default='0.0.0.0',
    required=False)
parser.add_argument(
    '-m',
    '--mlflow_port',
    type=int,
    help='port of mlflow tracking',
    default=5000,
    required=False)
parser.add_argument(
    '--device',
    type=str,
    help='device used for training',
    choices=['cpu', 'gpu'],
    default='cpu',
    required=False)
parser.add_argument(
    '--seed',
    type=int,
    help='seed for random number generators',
    default=58,
    required=False)
args = parser.parse_args()


# global variables
EVAL_MODE: EvalMode = EvalMode.CV
SEED: int = args.seed
VERBOSE = logging.DEBUG if args.verbose == 'debug' else logging.INFO
DEVICE: str = args.device

# configure MLFlow tracking remote server
#  see `mlflow-server.sh` for port and hostname. Since
#  we are running locally, we can use the default values.
mlflow.set_tracking_uri(f'http://{args.bind}:{args.mlflow_port}')

# set libs to log to our logging config
LIBRARIES = ['snorkel', 'vizard', 'flaml']

# Set up root logger, and add a file handler to root logger
MLFLOW_ARTIFACTS_BASE_PATH: Path = Path('artifacts')

# logging: setup progress bar
manager = enlighten.get_manager(sys.stderr)

# internal config handler
config_handler = JsonConfigHandler()

# path to source data, e.g. data.pkl file
PATH = args.dvc_data_path
# Git repo associated with DVC
REPO = args.dvc_repo
# use the latest EDA version (i.e. `vx.x.x-dev`)
VERSION = args.dvc_data_version

# metrics used for training snorkel model on unlabeled data
#   note that I don't want to move these into the `vizard` library as I believe
#   the developer who trains might want to play with these metrics, hence these constants
#   should be changable easily.
SNORKEL_LABEL_MODEL_METRICS: List[str] = ['accuracy', 'coverage', 'precision', 'recall', 'f1']
FLAML_AUTOML_METRICS: List[str] = ['accuracy', 'log_loss', 'f1', 'roc_auc']

# config the logging using our own custom logger which:
#   1. create artifact directory (for std log, mlflow, etc)
#   2. redirect logs of used libraries (including ours) to our std out
logger = loggers.Logger(
    name=__name__,
    level=VERBOSE,
    mlflow_artifacts_base_path=MLFLOW_ARTIFACTS_BASE_PATH,
    libs=LIBRARIES)
# create an instance of logging artifact
logger.create_artifact_instance()
# fitted transformation created by sklearn over our data for inference
MLFLOW_ARTIFACTS_MODELS_SKLEARN_TRANSFORM: Path = \
    logger.MLFLOW_ARTIFACTS_MODELS_PATH / 'train_sklearn_column_transfer.pkl'
# fitted model created by flaml automl ready for inference (and staging and production)
MLFLOW_ARTIFACTS_MODELS_FLAML_AUTOML: Path = \
    logger.MLFLOW_ARTIFACTS_MODELS_PATH / 'flaml_automl.pkl'


if __name__ == '__main__':
    try:
        
        logger.info('\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')
        # log experiment configs via mlflow
        MLFLOW_EXPERIMENT_NAME = f'{args.experiment_name if args.experiment_name else ""} - {VIZARD_VERSION}'
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run()

        logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
        logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')
        logger.info(f'DVC data version: {VERSION}')
        logger.info(f'DVC repo (root): {REPO}')
        logger.info(f'DVC data source path: {PATH}')
        logger.info('\t\t↑↑↑ Finished setting up configs: dirs, mlflow, and dvc. ↑↑↑')

        logger.info('\t\t↓↓↓ Starting loading preprocessed (EDA) data from DVC ↓↓↓')
        # get url data from DVC data storage
        data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
        # read dataset from remote (local) data storage
        data = pd.read_pickle(data_url)

        # get a copy of all data for extracting all categories for normalization
        #   for more information, see: issues #67 and #58
        data_all = data.copy(deep=True)

        logger.info(f'preprocessed data in raw PATH={PATH}'
                    f' with VERSION={VERSION},\n'
                    f'loaded from DVC storage at {data_url}.')
        logger.info('\t\t↑↑↑ Finished loading preprocessed (EDA) data from DVC ↑↑↑')

        # using snorkel for weak supervision to label the data
        logger.info('\t\t↓↓↓ Starting labeling data with snorkel ↓↓↓')
        logger.info('prepare data by separating already labeled (`acc` and `rej`)'
                    ' from weak and unlabeled data (`w-acc`, `w-rej` and `no idea`)')
        output_name = 'VisaResult'
        # for training the snorkel label model
        data_unlabeled = data[
            (data[output_name] != ClassificationLabels.ACC) &
            (data[output_name] != ClassificationLabels.REJ)].copy()
        # for testing the snorkel label model
        data_labeled = data[
            (data[output_name] == ClassificationLabels.ACC) |
            (data[output_name] == ClassificationLabels.REJ)].copy()
        logger.info(f'shape of unlabeled data: {data_unlabeled.shape}')
        logger.info(f'shape of labeled unlabeled data: {data_labeled.shape}')
        # convert strong to weak temporary to `lf_weak_*` so `LabelFunction`s'
        #   can work i.e. convert `acc` and `rej` in *labeled* dataset to `w-acc` and `w-rej`'
        data_labeled[output_name] = data_labeled[output_name].apply(
            lambda x: ClassificationLabels.WEAK_ACC if x == ClassificationLabels.ACC \
                else ClassificationLabels.WEAK_REJ)
        
        logger.info('\t↓↓↓ Starting extracting label matrices (L) by applying `LabelFunction`s ↓↓↓')
        # labeling functions
        lf_compose = [
            labeling.WeakAccept(),
            labeling.WeakReject(),
            labeling.NoIdea(),
        ]
        lfs = labeling.ComposeLFLabeling(labelers=lf_compose)()
        applier = PandasLFApplier(lfs)
        # apply LFs to the unlabeled (for `LabelModel` training) and
        #   labeled (for `LabelModel` test)
        label_matrix_train = applier.apply(data_unlabeled)
        # Remark: only should be used for evaluation of trained `LabelModel`
        #   and no where else
        label_matrix_test = applier.apply(data_labeled)

        y_test = data_labeled[output_name].apply(
            lambda x: labeling.ACC if x == ClassificationLabels.WEAK_ACC else labeling.REJ).values
        y_train = data_unlabeled[output_name].apply(
            lambda x: labeling.ACC if x == ClassificationLabels.WEAK_ACC else labeling.REJ).values
        # LF reports
        logger.info(LFAnalysis(L=label_matrix_train, lfs=lfs).lf_summary())
        logger.info('\t↑↑↑ Finishing extracting label matrices (L) by applying `LabelFunction`s ↑↑↑')
        
        logger.info('\t↓↓↓ Starting training `LabelModel` ↓↓↓')
        # train the label model and compute the training labels
        label_model_args = config_handler.parse(
            filename=LABEL_MODEL_CONFIGS,
            target='LabelModel'
        )
        config_handler.as_mlflow_artifact(
            logger.MLFLOW_ARTIFACTS_CONFIGS_PATH
        )
        logger.info(f'Training using device="{DEVICE}"')
        label_model = LabelModel(
            **label_model_args['method_init'],
            verbose=True,
            device=DEVICE
        )
        label_model.train()
        label_model.fit(
            label_matrix_train,
            **label_model_args['method_fit'],
            seed=SEED
        )
        logger.info('\t↑↑↑ Finished training LabelModel ↑↑↑')
        
        logger.info('\t↓↓↓ Starting inference on LabelModel ↓↓↓')
        # test the label model
        with torch.inference_mode():
            # predict labels for unlabeled data
            label_model.eval()
            auto_label_column_name = 'AL'
            logger.info(f'ModelLabel prediction is saved in "{auto_label_column_name}" column.')
            data_unlabeled.loc[:, auto_label_column_name] = label_model.predict(
                L=label_matrix_train,
                tie_break_policy='abstain'
            )
            # report train accuracy (train data here is our unlabeled data)
            modeling.report_label_model(
                label_model=label_model,
                label_matrix=label_matrix_train,
                gold_labels=y_train,
                metrics=SNORKEL_LABEL_MODEL_METRICS,
                set='train'
            )
            # report test accuracy (test data here is our labeled data which is larger (good!))
            label_model_metrics = modeling.report_label_model(
                label_model=label_model,
                label_matrix=label_matrix_test,
                gold_labels=y_test,
                metrics=SNORKEL_LABEL_MODEL_METRICS,
                set='test'
            )
            
            for m in SNORKEL_LABEL_MODEL_METRICS:
                mlflow.log_metric(
                    key=f'SnorkelLabelModel_{m}',
                    value=label_model_metrics[m]
                )
        logger.info('\t↑↑↑ Finishing inference on LabelModel ↑↑↑')
        # merge unlabeled data into all data
        data_unlabeled[auto_label_column_name] = data_unlabeled[auto_label_column_name].apply(
            lambda x: ClassificationLabels.ACC if x == labeling.ACC else 
            ClassificationLabels.REJ if x == labeling.REJ else ClassificationLabels.NO_IDEA)
        data.loc[data_unlabeled.index, [output_name]] = data_unlabeled[auto_label_column_name]
        data[output_name] = data[output_name].astype('object').astype('category')
        logger.info('\t\t↑↑↑ Finished labeling data with snorkel ↑↑↑')

        if EVAL_MODE == EvalMode.CV:
            pass
        else:
            # split to train and test to only augment train set
            pandas_train_test_splitter = preprocessors.PandasTrainTestSplit(
                random_state=SEED
            )

            data_tuple: Tuple[Any, ...] = pandas_train_test_splitter(
                df=data,
                target_column=output_name
            )
            data_train: pd.DataFrame = data_tuple[0]
            data_test: pd.DataFrame = data_tuple[1]
            data = data_train

            # dump json config into artifacts
            pandas_train_test_splitter.as_mlflow_artifact(
                logger.MLFLOW_ARTIFACTS_CONFIGS_PATH
            )

        logger.info('\t\t↓↓↓ Starting augmentation via snorkel (TFs) ↓↓↓')
        # transformation functions
        tf_compose = [
            augmentation.AddNormalNoiseDOBYear(dataframe=data),
            augmentation.AddNormalNoiseDateOfMarr(dataframe=data),
            augmentation.AddNormalNoiseOccRowXPeriod(dataframe=data, row=1),
            augmentation.AddNormalNoiseOccRowXPeriod(dataframe=data, row=2),
            augmentation.AddNormalNoiseOccRowXPeriod(dataframe=data, row=3),
            augmentation.AddNormalNoiseHLS(dataframe=data),
            augmentation.AddCategoricalNoiseSex(dataframe=data),
            augmentation.AddOrderedNoiseChdAccomp(dataframe=data, sec='B'),
            augmentation.AddOrderedNoiseChdAccomp(dataframe=data, sec='C')
        ]
        tfs = augmentation.ComposeTFAugmentation(augments=tf_compose)()  # type: ignore
        # define policy for applying TFs
        all_policy = ApplyAllPolicy(
            n_tfs=len(tfs), #sequence_length=len(tfs),
            n_per_original=1,  # TODO: #20
            keep_original=True
        )
        # apply TFs to all data (labels are not used, so no worries currently)
        tf_applier = PandasTFApplier(tfs, all_policy)
        data_augmented = tf_applier.apply(data)
        # TF reports
        logger.info(f'Original dataset size: {len(data)}')
        logger.info(f'Augmented dataset size: {len(data_augmented)}')
        logger.info('\t\t↑↑↑ Finishing augmentation via snorkel (TFs) ↑↑↑')

        logger.info('\t\t↓↓↓ Starting preprocessing on directly DVC `vX.X.X-dev` data ↓↓↓')
        # change dtype of augmented data to be as original data
        data_augmented = data_augmented.astype(data.dtypes)
        # use augmented data from now on
        data = data_augmented
        # move the dependent variable to the end of the dataframe
        data = preprocessors.move_dependent_variable_to_end(
            df=data,
            target_column=output_name
        )

        # convert to np and split to train, test, eval
        y_train = data[output_name].to_numpy()
        x_train = data.drop(columns=[output_name], inplace=False).to_numpy()

        if EVAL_MODE == EvalMode.CV:
            pass
        else:
            train_test_eval_splitter = preprocessors.TrainTestEvalSplit(
                random_state=SEED
            )
            data_tuple = train_test_eval_splitter(
                df=data_test,
                target_column=output_name
            )
            x_test, x_eval, y_test, y_eval = data_tuple
            # dump json config into artifacts
            train_test_eval_splitter.as_mlflow_artifact(
                logger.MLFLOW_ARTIFACTS_CONFIGS_PATH
            )

        # Transform and normalize appropriately given config
        x_column_transformers_config = preprocessors.ColumnTransformerConfig()
        x_column_transformers_config.set_configs(
            preprocessors.CANADA_COLUMN_TRANSFORMER_CONFIG_X
        )
        # dump json config into artifacts
        x_column_transformers_config.as_mlflow_artifact(
            logger.MLFLOW_ARTIFACTS_CONFIGS_PATH
        )

        x_ct = preprocessors.ColumnTransformer(
            transformers=x_column_transformers_config.generate_pipeline(
                df=data,
                df_all=data_all
            ),
            remainder='passthrough',
            verbose=False,
            verbose_feature_names_out=False,
            n_jobs=None,
        )
        y_ct = preprocessors.LabelBinarizer()
        # fit and transform on train data
        xt_train = x_ct.fit_transform(x_train)  # TODO: see #41, #42
        yt_train = y_ct.fit_transform(y_train)  # TODO: see #47, #42
        # save the fitted transforms as artifacts for later use
        with open(MLFLOW_ARTIFACTS_MODELS_SKLEARN_TRANSFORM, 'wb') as f:
            pickle.dump(x_ct, f, pickle.HIGHEST_PROTOCOL)
        
        if EVAL_MODE == EvalMode.CV:
            pass
        else:
            # transform on eval data
            xt_eval = x_ct.transform(x_eval)
            yt_eval = y_ct.transform(y_eval)
            # transform on test data
            xt_test = x_ct.transform(x_test)
            yt_test = y_ct.transform(y_test)

        # preview the transformed data
        preview_ct = preprocessors.preview_column_transformer(
            column_transformer=x_ct,
            original=x_train,
            transformed=xt_train,
            df=data,
            random_state=SEED,
            n_samples=1
        )
        logger.info([_ for _ in preview_ct])
        logger.info('\t\t↑↑↑ Finished preprocessing on directly DVC `vX.X.X-dev` data ↑↑↑')

        logger.info('\t\t↓↓↓ Starting defining estimators models ↓↓↓')
        flaml_automl = trainers.AutoML()
        flaml_automl_args = config_handler.parse(
            filename=trainers.FLAML_AUTOML_CONFIGS,
            target='FLAML_AutoML'
        )
        config_handler.as_mlflow_artifact(
            logger.MLFLOW_ARTIFACTS_CONFIGS_PATH
        )
        logger.info('\t\t↑↑↑ Finished defining estimators models ↑↑↑')

        logger.info('\t\t↓↓↓ Starting loading training config and training estimators ↓↓↓')
        flaml_automl.fit(
            X_train=xt_train,
            y_train=yt_train,
            X_val=None if EVAL_MODE==EvalMode.CV else xt_eval,
            y_val=None if EVAL_MODE==EvalMode.CV else yt_eval,
            eval_method=EVAL_MODE,
            seed=SEED,
            append_log=False,
            log_file_name=logger.MLFLOW_ARTIFACTS_LOGS_PATH / 'flaml.log',
            **flaml_automl_args['method_fit'])
        # report feature importance
        feature_names = preprocessors.get_transformed_feature_names(
            column_transformer=x_ct,
            original_columns_names=data.drop(columns=[output_name]).columns.values,
        )
        logger.info(
            trainers.report_feature_importances(
                estimator=flaml_automl.model.estimator,
                feature_names=feature_names
            )
        )

        if EVAL_MODE == EvalMode.CV:
            pass
        else:
            y_pred = flaml_automl.predict(xt_test)
            logger.info(f'Best FLAML model: {flaml_automl.model.estimator}')
            metrics_loss_score_dict = trainers.get_loss_score(
                y_predict=y_pred,
                y_true=yt_test,
                metrics=FLAML_AUTOML_METRICS
            )
            logger.info(trainers.report_loss_score(metrics=metrics_loss_score_dict))

        # Save the model
        with open(MLFLOW_ARTIFACTS_MODELS_FLAML_AUTOML, 'wb') as f:
            pickle.dump(flaml_automl, f, pickle.HIGHEST_PROTOCOL)
        # track (and register) the model via mlflow flavors
        trainers.aml_flaml.log_model(
            estimator=flaml_automl,
            artifact_path='/'.join(logger.MLFLOW_ARTIFACTS_MLMODELS_PATH.parts[1:]),
            conda_env='conda_env.yml',
            registered_model_name=None  # manually register desired models
        )
        logger.info('\t\t↑↑↑ Finished loading training config and training estimators ↑↑↑')

    except Exception as e:
        logger.error(e)
    
    # cleanup code
    finally:
        # Log artifacts (logs, saved files, etc)
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_BASE_PATH)
        # delete redundant logs, files that are logged as artifact
        shutil.rmtree(MLFLOW_ARTIFACTS_BASE_PATH)

        logger.info('\t\t↓↓↓ Starting logging hyperparams and params with MLFlow ↓↓↓')
        logger.info('Log global params')
        mlflow.log_param('device', DEVICE)
        # log data params
        logger.info('Log EDA data params as MLflow params...')
        mlflow.log_param('EDA_dataset_dir', PATH)
        mlflow.log_param('EDA_data_url', data_url)
        mlflow.log_param('EDA_data_version', VERSION)
        mlflow.log_param('EDA_input_shape', data.shape)
        mlflow.log_param('EDA_input_columns', data.columns.values)
        mlflow.log_param('EDA_input_dtypes', data.dtypes.values)
        # LabelModel params
        logger.info('Log Snorkel `LabelModel` params as MLflow params...')
        mlflow.log_param('LabelModel_fit_method', label_model_args['method_fit'])
        mlflow.log_param('labeled_dataframe_shape', data_labeled.shape)
        mlflow.log_param('unlabeled_dataframe_shape', data_unlabeled.shape)
        # log FLAML AutoML params
        logger.info('Log `FLAML` `AutoML` params as MLflow params...')
        if EVAL_MODE == EvalMode.CV:
            pass
        else:
            mlflow.log_metrics(metrics_loss_score_dict)
        # log modeling preprocessed params
        logger.info('Log modeling preprocessed params as MLflow params...')
        mlflow.log_param('x_train_shape', x_train.shape)
        mlflow.log_param('xt_train_shape', xt_train.shape)
        mlflow.log_param('y_train_shape', y_train.shape)
        mlflow.log_param('yt_train_shape', yt_train.shape)
        if EVAL_MODE == EvalMode.CV:
            pass
        else:
            mlflow.log_param('x_test_shape', x_test.shape)
            mlflow.log_param('x_val_shape', x_eval.shape)
            mlflow.log_param('y_test_shape', y_test.shape)
            mlflow.log_param('y_val_shape', y_test.shape)
            logger.info('\t\t↑↑↑ Finished logging hyperparams and params with MLFlow ↑↑↑')
        mlflow.end_run()
