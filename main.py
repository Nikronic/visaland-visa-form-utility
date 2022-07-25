# core
import pandas as pd
# ours: data
from vizard.models import preprocessors
from vizard.version import VERSION as VIZARD_VERSION
# devops
import dvc.api
import mlflow
# helpers
from pathlib import Path
import enlighten
import logging
import shutil
import sys
import os



if __name__ == '__main__':

    # globals
    SEED = 322
    VERBOSE = logging.DEBUG

    # configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(VERBOSE)

    # Set up root logger, and add a file handler to root logger
    MLFLOW_ARTIFACTS_PATH = Path('artifacts')
    MLFLOW_ARTIFACTS_LOGS_PATH = MLFLOW_ARTIFACTS_PATH / 'logs'
    MLFLOW_ARTIFACTS_CONFIGS_PATH = MLFLOW_ARTIFACTS_PATH / 'configs'
    if not os.path.exists(MLFLOW_ARTIFACTS_PATH):
        os.makedirs(MLFLOW_ARTIFACTS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_LOGS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_CONFIGS_PATH)

    logger_handler = logging.FileHandler(filename=MLFLOW_ARTIFACTS_LOGS_PATH / 'main.log',
                                        mode='w')
    logger.addHandler(logger_handler)  # type: ignore
    # set libs to log to our logging config
    __libs = ['snorkel', 'vizard']
    for __l in __libs:
        __libs_logger = logging.getLogger(__l)
        __libs_logger.setLevel(VERBOSE)
        __libs_logger.addHandler(logger_handler)
    # logging: setup progress bar
    manager = enlighten.get_manager(sys.stderr)  

    try:
        logger.info(
            '\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓')
        # main path
        SRC_DIR = '/mnt/e/dataset/processed/all/'  # path to source encrypted pdf
        DST_DIR = 'raw-dataset/all/'  # path to decrypted pdf

        # data versioning config
        PATH = DST_DIR[:-1] + '-dev.pkl'  # path to source data, e.g. data.pkl file
        REPO = '/home/nik/visaland-visa-form-utility'
        VERSION = 'v1.2.2-dev'  # use the latest EDA version (i.e. `vx.x.x-dev`)

        # log experiment configs
        MLFLOW_EXPERIMENT_NAME = f'full pipelines - {VIZARD_VERSION}'
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        # VIZARD_VERSION is used to differentiate states of progress of
        #  FULL pipeline implementation.
        # Since I update version of the pipeline every time I make a considerable
        #  change, using this for MLflow experiment name would help to identify
        #  the state of the pipeline and its issues.
        MLFLOW_TAGS = {
            'stage': 'dev'  # dev, beta, production
        }
        mlflow.set_tags(MLFLOW_TAGS)

        logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
        logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')
        logger.info(f'DVC data version: {VERSION}')
        logger.info(f'DVC repo (root): {REPO}')
        logger.info(f'DVC data source path: {PATH}')
        logger.info(
            '\t\t↑↑↑ Finished setting up configs: dirs, mlflow, dvc, etc ↑↑↑')

        logger.info('\t\t↓↓↓ Starting loading preprocessed (EDA) data from DVC ↓↓↓')
        # get url data from DVC data storage
        data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
        # read dataset from remote (local) data storage
        data = pd.read_pickle(data_url)
        logger.info(f'preprocessed data in raw PATH={PATH}'
                    f' with VERSION={VERSION}, loaded from',
                    f' DVC storage at {data_url}.')
        logger.info('\t\t↑↑↑ Finished loading preprocessed (EDA) data from DVC ↑↑↑')

        logger.info('\t\t↓↓↓ Starting preprocessing on directly DVC `vX.X.X-dev` data ↓↓↓')
        # TODO: add preprocessing steps here

        # move the dependent variable to the end of the dataframe
        data = preprocessors.move_dependent_variable_to_end(
            df=data, target_column='VisaResult')

        # convert to np and split to train, test, eval
        train_test_eval_splitter = preprocessors.TrainTestEvalSplit(random_state=SEED)
        data_tuple = train_test_eval_splitter(df=data, target_column='VisaResult')
        x_train, x_test, x_eval, y_train, y_test, y_eval = data_tuple
        # dump json config into artifacts
        train_test_eval_splitter.as_mlflow_artifact(MLFLOW_ARTIFACTS_CONFIGS_PATH)

        # Transform and normalize appropriately given config
        x_column_transformers_config = preprocessors.ColumnTransformerConfig()
        x_column_transformers_config.set_configs(
            preprocessors.CANADA_COLUMN_TRANSFORMER_CONFIG_X)
        # dump json config into artifacts
        x_column_transformers_config.as_mlflow_artifact(MLFLOW_ARTIFACTS_CONFIGS_PATH)

        x_ct = preprocessors.ColumnTransformer(
            transformers=x_column_transformers_config.generate_pipeline(df=data),
            remainder='passthrough',
            verbose=False,
            verbose_feature_names_out=False,
            n_jobs=None,
        )
        y_ct = preprocessors.LabelBinarizer()
        # fit and transform on train data
        xt_train = x_ct.fit_transform(x_train)  # TODO: see #53
        yt_train = y_ct.fit_transform(y_train)  # TODO: see #54
        # transform on eval data
        xt_eval = x_ct.transform(x_eval)
        yt_eval = y_ct.transform(y_eval)
        # transform on test data
        xt_test = x_ct.transform(x_test)
        yt_test = y_ct.transform(y_test)

        # preview the transformed data
        preview_ct = preprocessors.preview_column_transformer(column_transformer=x_ct,
                                                            original=x_train,
                                                            transformed=xt_train,
                                                            df=data,
                                                            random_state=SEED,
                                                            n_samples=1)
        logger.info([_ for _ in preview_ct])

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
        # TODO: log preprocessor configs
        # TODO: log estimator params
        # TODO: log trainer config
        # TODO: log evaluator config
        # TODO: log weights
        # TODO: log anything else in between that needs to be logged
        # log data params
        logger.info('Log EDA data params as MLflow params...')
        mlflow.log_param('EDA_dataset_dir', DST_DIR)
        mlflow.log_param('EDA_data_url', data_url)
        mlflow.log_param('EDA_data_version', VERSION)
        mlflow.log_param('EDA_input_shape', data.shape)
        mlflow.log_param('EDA_input_columns', data.columns.values)
        mlflow.log_param('EDA_input_dtypes', data.dtypes.values)
        # log modeling preprocessed params
        logger.info('Log modeling preprocessed params as MLflow params...')
        mlflow.log_param('x_train_shape', x_train.shape)
        mlflow.log_param('xt_train_shape', xt_train.shape)
        mlflow.log_param('x_test_shape', x_test.shape)
        mlflow.log_param('x_val_shape', x_eval.shape)
        mlflow.log_param('y_train_shape', y_train.shape)
        mlflow.log_param('yt_train_shape', yt_train.shape)
        mlflow.log_param('y_test_shape', y_test.shape)
        mlflow.log_param('y_val_shape', y_test.shape)
        logger.info('\t\t↑↑↑ Finished logging with MLFlow ↑↑↑')

    except Exception as e:
        logger.error(e)
        # Log artifacts (logs, saved files, etc)
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_PATH)
        # delete redundant logs, files that are logged as artifact
        shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
