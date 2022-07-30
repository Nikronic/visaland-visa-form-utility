# core
import pandas as pd
import torch
# ours: snorkel
from vizard.snorkel import LABEL_MODEL_CONFIGS
from vizard.snorkel import labeling
from vizard.snorkel import modeling
from vizard.snorkel import augmentation
from vizard.snorkel import slicing
from vizard.snorkel import PandasLFApplier
from vizard.snorkel import PandasTFApplier
from vizard.snorkel import PandasSFApplier
from vizard.snorkel import LFAnalysis
from vizard.snorkel import LabelModel
from vizard.snorkel import ApplyAllPolicy
from vizard.snorkel import Scorer
from vizard.snorkel import preview_tfs
from vizard.snorkel import slice_dataframe
# ours: models
from vizard.models import preprocessors
from vizard.models import trainers
# ours: helpers
from vizard.version import VERSION as VIZARD_VERSION
from vizard.configs import JsonConfigHandler
# devops
import dvc.api
import mlflow
# helpers
from pathlib import Path
import enlighten
import logging
import pickle
import shutil
import sys
import os



if __name__ == '__main__':

    # globals
    SEED = 322
    VERBOSE = logging.DEBUG
    DEVICE = 'cuda'

    # configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(VERBOSE)
    logger_formatter = logging.Formatter(
        "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
    )

    # Set up root logger, and add a file handler to root logger
    MLFLOW_ARTIFACTS_PATH = Path('artifacts')
    MLFLOW_ARTIFACTS_LOGS_PATH = MLFLOW_ARTIFACTS_PATH / 'logs'
    MLFLOW_ARTIFACTS_CONFIGS_PATH = MLFLOW_ARTIFACTS_PATH / 'configs'
    MLFLOW_ARTIFACTS_WEIGHTS_PATH = MLFLOW_ARTIFACTS_PATH / 'weights'
    if not os.path.exists(MLFLOW_ARTIFACTS_PATH):
        os.makedirs(MLFLOW_ARTIFACTS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_LOGS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_CONFIGS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_WEIGHTS_PATH)

    logger_handler = logging.FileHandler(filename=MLFLOW_ARTIFACTS_LOGS_PATH / 'main.log',
                                        mode='w')
    stdout_stream_handler = logging.StreamHandler(stream=sys.stdout)
    stderr_stream_handler = logging.StreamHandler(stream=sys.stderr)
    logger_handler.setFormatter(logger_formatter)
    stdout_stream_handler.setFormatter(logger_formatter)
    stderr_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)  # type: ignore
    logger.addHandler(stdout_stream_handler)
    logger.addHandler(stderr_stream_handler)
    
    # set libs to log to our logging config
    __libs = ['snorkel', 'vizard', 'flaml']
    for __l in __libs:
        __libs_logger = logging.getLogger(__l)
        __libs_logger.setLevel(VERBOSE)
        __libs_logger.addHandler(logger_handler)
        __libs_logger.addHandler(stdout_stream_handler)
        __libs_logger.addHandler(stderr_stream_handler)
    # logging: setup progress bar
    manager = enlighten.get_manager(sys.stderr)

    # internal config handler
    config_handler = JsonConfigHandler()

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
        MLFLOW_EXPERIMENT_NAME = f'Fix #61 - FLAML AutoML - full pipelines - {VIZARD_VERSION}'
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
        logger.info('\t\t↑↑↑ Finished setting up configs: dirs, mlflow, and dvc. ↑↑↑')

        logger.info('\t\t↓↓↓ Starting loading preprocessed (EDA) data from DVC ↓↓↓')
        # get url data from DVC data storage
        data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
        # read dataset from remote (local) data storage
        data = pd.read_pickle(data_url)

        #################  # TODO: fix DATA
        z = data.isna().sum() != 0 
        data.iloc[:, z.values] = data.iloc[:, z.values].fillna(value=85, inplace=False)
        #################  # TODO: fix DATA

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
        data_unlabeled = data[(data[output_name] != 'acc') &
                        (data[output_name] != 'rej')].copy()
        # for testing the snorkel label model
        data_labeled = data[(data[output_name] == 'acc') |
                        (data[output_name] == 'rej')].copy()
        logger.info(f'shape of unlabeled data: {data_unlabeled.shape}')
        logger.info(f'shape of labeled unlabeled data: {data_labeled.shape}')
        # convert strong to weak temporary to `lf_weak_*` so `LabelFunction`s'
        #   can work i.e. convert `acc` and `rej` in *labeled* dataset to `w-acc` and `w-rej`'
        data_labeled[output_name] = data_labeled[output_name].apply(
            lambda x: 'w-acc' if x == 'acc' else 'w-rej')
        
        logger.info('\t↓↓↓ Starting extracting label matrices (L) by applying `LabelFunction`s ↓↓↓')
        # labeling functions
        lf_compose = [
            labeling.WeakAccept(),
            labeling.WeakReject(),
            labeling.NoIdea(),
        ]
        lfs = labeling.ComposeLFLabeling(labelers=lf_compose)()
        applier = PandasLFApplier(lfs)
        # apply LFs to the unlabeled (for `LabelModel` training) and labeled (for `LabelModel` test)
        label_matrix_train = applier.apply(data_unlabeled)
        # Remark: only should be used for evaluation of trained `LabelModel` and no where else
        label_matrix_test = applier.apply(data_labeled)

        y_test = data_labeled[output_name].apply(
            lambda x: labeling.ACC if x == 'w-acc' else labeling.REJ).values
        y_train = data_unlabeled[output_name].apply(
            lambda x: labeling.ACC if x == 'w-acc' else labeling.REJ).values
        # LF reports
        logger.info(LFAnalysis(L=label_matrix_train, lfs=lfs).lf_summary())
        logger.info('\t↑↑↑ Finishing extracting label matrices (L) by applying `LabelFunction`s ↑↑↑')
        
        logger.info('\t↓↓↓ Starting training `LabelModel` ↓↓↓')
        # train the label model and compute the training labels
        label_model_args = config_handler.parse(filename=LABEL_MODEL_CONFIGS,
                                                target='LabelModel')
        config_handler.as_mlflow_artifact(MLFLOW_ARTIFACTS_CONFIGS_PATH)
        logger.info(f'Training using device="{DEVICE}"')
        label_model = LabelModel(**label_model_args['method_init'],
                                 verbose=True, device=DEVICE)
        label_model.train()
        label_model.fit(label_matrix_train,
                        **label_model_args['method_fit'],
                        seed=SEED)
        logger.info('\t↑↑↑ Finished training LabelModel ↑↑↑')
        
        logger.info('\t↓↓↓ Starting inference on LabelModel ↓↓↓')
        # test the label model
        with torch.inference_mode():
            # predict labels for unlabeled data
            label_model.eval()
            auto_label_column_name = 'AL'
            logger.info(f'ModelLabel prediction is saved in "{auto_label_column_name}" column.')
            data_unlabeled.loc[:, auto_label_column_name] = label_model.predict(
                L=label_matrix_train, tie_break_policy='abstain')
            # report train accuracy (train data here is our unlabeled data)
            metrics = ['accuracy', 'coverage', 'precision', 'recall', 'f1']
            modeling.report_label_model(label_model=label_model,
                                        label_matrix=label_matrix_train,
                                        gold_labels=y_train,
                                        metrics=metrics,
                                        set='train')
            # report test accuracy (test data here is our labeled data which is larger (good!))
            label_model_metrics = modeling.report_label_model(label_model=label_model,
                                                              label_matrix=label_matrix_test,
                                                              gold_labels=y_test,
                                                              metrics=metrics,
                                                              set='test')
            
            for m in metrics:
                mlflow.log_metric(key=f'SnorkelLabelModel_{m}',
                                  value=label_model_metrics[m])
        logger.info('\t↑↑↑ Finishing inference on LabelModel ↑↑↑')
        # merge unlabeled data into all data
        data_unlabeled[auto_label_column_name] = data_unlabeled[auto_label_column_name].apply(
            lambda x: 'acc' if x == labeling.ACC else 
            'rej' if x == labeling.REJ else 'no idea')
        data.loc[data_unlabeled.index, [output_name]] = data_unlabeled[auto_label_column_name]
        data[output_name] = data[output_name].astype('object').astype('category')
        logger.info('\t\t↑↑↑ Finished labeling data with snorkel ↑↑↑')

        logger.info('\t\t↓↓↓ Starting augmentation via snorkel (TFs) ↓↓↓')
        # transformation functions
        tf_compose = [
            augmentation.AddOrderedNoiseChdAccomp(dataframe=data, sec='B'),
            augmentation.AddOrderedNoiseChdAccomp(dataframe=data, sec='C')
        ]
        tfs = augmentation.ComposeTFAugmentation(augments=tf_compose)()
        # define policy for applying TFs
        all_policy = ApplyAllPolicy(n_tfs=len(tfs), #sequence_length=len(tfs),
                                    n_per_original=2,  # TODO: #20
                                    keep_original=True)
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
        logger.info('\t\t↑↑↑ Finishing augmentation via snorkel (TFs) ↑↑↑')

        logger.info('\t\t↓↓↓ Starting slicing by snorkel (SFs) ↓↓↓')
        # slicing functions
        sf_compose = [
            slicing.SinglePerson(),
        ]
        sfs = slicing.ComposeSFSlicing(slicers=sf_compose)()
        single_person_slice = slice_dataframe(data_augmented, sfs[0])
        logger.info(single_person_slice.sample(5))
        sf_applier = PandasSFApplier(sfs)
        data_augmented_sliced = sf_applier.apply(data_augmented)
        scorer = Scorer(metrics=metrics)
        # TODO: use slicing `scorer` only for `test` set
        # logger.info(scorer.score_slices(S=S_test, golds=Y_test,
        #             preds=preds_test, probs=probs_test, as_dataframe=True))
        logger.info('\t\t↑↑↑ Finishing slicing by snorkel (SFs) ↑↑↑')

        logger.info('\t\t↓↓↓ Starting preprocessing on directly DVC `vX.X.X-dev` data ↓↓↓')
        # TODO: add preprocessing steps here

        # change dtype of augmented data to be as original data
        data_augmented = data_augmented.astype(data.dtypes)
        # use augmented data from now on
        data = data_augmented
        # move the dependent variable to the end of the dataframe
        data = preprocessors.move_dependent_variable_to_end(
            df=data, target_column=output_name)

        # convert to np and split to train, test, eval
        train_test_eval_splitter = preprocessors.TrainTestEvalSplit(random_state=SEED)
        data_tuple = train_test_eval_splitter(df=data, target_column=output_name)
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
        flaml_automl = trainers.AutoML()
        flaml_automl_args = config_handler.parse(filename=trainers.FLAML_AUTOML_CONFIGS,
                                                 target='FLAML_AutoML')
        config_handler.as_mlflow_artifact(MLFLOW_ARTIFACTS_CONFIGS_PATH)
        flaml_automl.fit(X_train=xt_train, y_train=yt_train,
                         X_val=xt_eval, y_val=yt_eval, seed=SEED,
                         append_log=False,
                         log_file_name=MLFLOW_ARTIFACTS_LOGS_PATH / 'flaml.log',
                         **flaml_automl_args['method_fit'])
        y_pred = flaml_automl.predict(xt_test)
        logger.info(f'Best FLAML model: {flaml_automl.model.estimator}')
        metrics = ['accuracy', 'log_loss', 'f1', 'roc_auc', ]
        metrics_loss_score_dict = trainers.get_loss_score(y_predict=y_pred,
                                                          y_true=yt_test,
                                                          metrics=metrics)
        logger.info(trainers.report_loss_score(metrics=metrics_loss_score_dict))
        # Save the model
        with open(MLFLOW_ARTIFACTS_WEIGHTS_PATH / 'flaml_automl.pkl', 'wb') as f:
            pickle.dump(flaml_automl, f, pickle.HIGHEST_PROTOCOL)
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

    except Exception as e:
        logger.error(e)
    
    # cleanup code
    finally:
        # Log artifacts (logs, saved files, etc)
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_PATH)
        # delete redundant logs, files that are logged as artifact
        shutil.rmtree(MLFLOW_ARTIFACTS_PATH)

        logger.info('\t\t↓↓↓ Starting logging hyperparams and params with MLFlow ↓↓↓')
        logger.info('Log global params')
        mlflow.log_param('device', DEVICE)
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
        # LabelModel params
        logger.info('Log Snorkel `LabelModel` params as MLflow params...')
        mlflow.log_param('LabelModel_fit_method', label_model_args['method_fit'])
        mlflow.log_param('labeled_dataframe_shape', data_labeled.shape)
        mlflow.log_param('unlabeled_dataframe_shape', data_unlabeled.shape)
        # log FLAML AutoML params
        logger.info('Log `FLAML` `AutoML` params as MLflow params...')
        mlflow.log_metrics(metrics_loss_score_dict)
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
        logger.info('\t\t↑↑↑ Finished logging hyperparams and params with MLFlow ↑↑↑')
