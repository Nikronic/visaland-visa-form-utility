"""
Only run this code for generating [dataset_name].pkl file, every time we want to create a new version
of our dataset, then after this step, we use DVC to version it.
In main.py, we just rerun this step but integrated into MLFlow to track which version we are
USING (than generating).

In simple terms, if you added new samples, changed columns or anything that should be considered
permanent at the time, you should run this script, then version it with DVC and for doing
data analysis or machine learning for prediction, only pull from DVC remote storage of 
this version (or any version you want).
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import enlighten
import mlflow
import pandas as pd

from vizard.data import functional
from vizard.data.constant import DOC_TYPES
from vizard.data.preprocessor import (CanadaDataframePreprocessor, CopyFile,
                                      FileTransformCompose,
                                      MakeContentCopyProtectedMachineReadable)

if __name__ == "__main__":
    # globals
    SEED = 322
    VERBOSE = logging.DEBUG

    # configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(VERBOSE)

    # Set up root logger, and add a file handler to root logger
    MLFLOW_ARTIFACTS_PATH = Path("artifacts")
    MLFLOW_ARTIFACTS_LOGS_PATH = MLFLOW_ARTIFACTS_PATH / "logs"
    MLFLOW_ARTIFACTS_CONFIGS_PATH = MLFLOW_ARTIFACTS_PATH / "configs"
    if not os.path.exists(MLFLOW_ARTIFACTS_PATH):
        os.makedirs(MLFLOW_ARTIFACTS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_LOGS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_CONFIGS_PATH)

    logger_handler = logging.FileHandler(
        filename=MLFLOW_ARTIFACTS_LOGS_PATH / "import-data.log", mode="w"
    )
    logger.parent.addHandler(logger_handler)  # type: ignore
    # set libs to log to our logging config
    __libs = ["snorkel", "vizard"]
    for __l in __libs:
        __libs_logger = logging.getLogger(__l)
        __libs_logger.setLevel(VERBOSE)
        __libs_logger.addHandler(logger_handler)

    manager = enlighten.get_manager(sys.stderr)  # setup progress bar

    logger.info("\t\t↓↓↓ Starting setting up configs: dirs, mlflow, dvc, etc ↓↓↓")
    # main path
    SRC_DIR = "/mnt/e/dataset/processed/all/"  # path to source encrypted pdf
    DST_DIR = "raw-dataset/all/"  # path to decrypted pdf

    # MLFlow configs
    # data versioning config
    PATH = DST_DIR[:-1] + ".pkl"  # path to source data, e.g. data.pkl file
    REPO = "/home/nik/visaland-visa-form-utility"
    # the version it is GOING TO BE (the version you gonna `dvc add` and `git tag`)
    VERSION = "v1.0.3"
    # run `git tag` and use a newer version than anything with this pattern `vx.x.x` (without `-field-*`)

    # log experiment configs
    MLFLOW_EXPERIMENT_NAME = "full dataset construction from raw documents to csv"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    MLFLOW_TAGS = {"stage": "dev"}  # dev, beta, production
    mlflow.set_tags(MLFLOW_TAGS)

    logger.info(f"MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}")
    logger.info(f"MLflow experiment id: {mlflow.active_run().info.run_id}")
    logger.info(f"DVC data version: {VERSION}")
    logger.info(f"DVC repo (root): {REPO}")
    logger.info(f"DVC data source path: {PATH}")
    logger.info("\t\t↑↑↑ Finished setting up configs: dirs, mlflow, dvc, etc ↑↑↑")

    # main code
    logger.info("\t\t↓↓↓ Starting data extraction ↓↓↓")
    # Canada protected PDF to machine readable for all entries and transferring other files as it is
    compose = {
        CopyFile(mode="cf"): ".csv",
        CopyFile(mode="cf"): ".txt",
        MakeContentCopyProtectedMachineReadable(): ".pdf",
    }
    file_transform_compose = FileTransformCompose(transforms=compose)
    functional.process_directory(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        compose=file_transform_compose,
        file_pattern="*",
    )
    logger.info("\t\t↑↑↑ Finished data extraction ↑↑↑")

    logger.info("\t\t↓↓↓ Starting data preprocessing ↓↓↓")
    # convert PDFs to pandas dataframes
    SRC_DIR = DST_DIR[:-1]
    dataframe = pd.DataFrame()
    progress_bar = manager.counter(
        total=len(next(os.walk(DST_DIR), (None, [], None))[1]),
        desc="Preprocessed",
        unit="data point",
    )
    i = 0  # for progress bar
    for dirpath, dirnames, all_filenames in os.walk(SRC_DIR):
        dataframe_entry = pd.DataFrame()

        # filter all_filenames
        filenames = all_filenames
        if filenames:
            files = [os.path.join(dirpath, fname) for fname in filenames]
            # applicant form
            in_fname = [f for f in files if "5257" in f][0]
            df_preprocessor = CanadaDataframePreprocessor()
            if len(in_fname) != 0:
                dataframe_applicant = df_preprocessor.file_specific_basic_transform(
                    path=in_fname, type=DOC_TYPES.canada_5257e
                )
            # applicant family info
            in_fname = [f for f in files if "5645" in f][0]
            if len(in_fname) != 0:
                dataframe_family = df_preprocessor.file_specific_basic_transform(
                    path=in_fname, type=DOC_TYPES.canada_5645e
                )
            # manually added labels
            in_fname = [f for f in files if "label" in f][0]
            if len(in_fname) != 0:
                dataframe_label = df_preprocessor.file_specific_basic_transform(
                    path=in_fname, type=DOC_TYPES.canada_label
                )

            # final dataframe: concatenate common forms and label column wise
            dataframe_entry = pd.concat(
                objs=[dataframe_applicant, dataframe_family, dataframe_label],
                axis=1,
                verify_integrity=True,
            )

        # concat the dataframe_entry into the main dataframe (i.e. adding rows)
        dataframe = pd.concat(
            objs=[dataframe, dataframe_entry],
            axis=0,
            verify_integrity=True,
            ignore_index=True,
        )
        # logging
        i += 1
        logger.info(f"Processed {i}th data point...")
        progress_bar.update()

    # save dataframe to disc as pickle
    logger.info("\t↓↓↓ Starting saving dataframe to disc ↓↓↓")
    dataset_path = DST_DIR[:-1] + ".pkl"
    dataframe.to_pickle(dataset_path)
    logger.info(f"Dataframe saved to path={dataset_path}")
    logger.info("\t↑↑↑ Finished saving dataframe to disc ↑↑↑")
    logger.info("\t\t↑↑↑ Finished data preprocessing ↑↑↑")

    # log data params
    logger.info("\t\t↓↓↓ Starting logging with MLFlow ↓↓↓")
    mlflow.log_param("raw_dataset_dir", DST_DIR)
    mlflow.log_param("data_version", VERSION)
    mlflow.log_param("input_shape", dataframe.shape)
    mlflow.log_param("input_columns", dataframe.columns.values)
    mlflow.log_param("input_dtypes", dataframe.dtypes.values)
    logger.info("\t\t↑↑↑ Finished logging with MLFlow ↑↑↑")

    # Log artifacts (logs, saved files, etc)
    mlflow.log_artifacts(MLFLOW_ARTIFACTS_PATH)
    # delete redundant logs, files that are logged as artifact
    shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
