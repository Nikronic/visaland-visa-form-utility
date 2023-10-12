# syntax=docker/dockerfile:1

FROM condaforge/mambaforge AS build

WORKDIR /visaland-visa-form-utility

RUN mamba create --name viz-inf-cpu python=3.8.0 -y

# Make all RUN commands use the conda env
SHELL ["mamba", "run", "-n", "viz-inf-cpu", "/bin/bash", "-c"]

# update pip
RUN pip install --upgrade pip

# Pin the Python version
RUN echo "python 3.8.0" >> $CONDA_PREFIX/conda-meta/pinned

# install mlflow
RUN pip install mlflow==1.28.0
# Install data extraction dependencies
RUN mamba install -c conda-forge xmltodict=0.13.0 -y
RUN pip install pikepdf==5.1.5
RUN pip install pypdf2==2.2.1
# Install PyTorch
RUN mamba install pytorch==1.10.1 cpuonly -c pytorch -y
# Install enlighten for beauty
RUN pip install enlighten==1.10.2
# Install ML libs
RUN mamba install xgboost==1.3.3 -y
RUN mamba install -c conda-forge lightgbm==3.3.3 -y
RUN pip install catboost==1.0.6
RUN mamba install -c conda-forge flaml=1.0.12 -y
RUN mamba install -c conda-forge shap=0.41.0 -y
RUN pip install shapely==1.8.5.post1
# Fix broken dependencies
RUN mamba install -c conda-forge numba=0.56.3 -y
RUN mamba install -c conda-forge numpy=1.24.4 -y
# Install API libs
RUN pip install pydantic==1.9.1
RUN pip install fastapi==0.85.0
RUN pip install uvicorn==0.18.2
# Data Version Control (DVC) tool
RUN mamba install -c conda-forge dvc=2.10.2 -y
RUN mamba install -c conda-forge dvclive=0.12.1 -y
# Again some clueless broken dependencies
RUN mamba install -c conda-forge fsspec=2022.10.0 -y
RUN mamba install -c conda-forge funcy=1.17 -y
# Install ray distributed training
RUN pip install ray==1.13.0
# Install custom fork of Snorkel
COPY snorkel-0.9.8/ /visaland-visa-form-utility/snorkel-0.9.8/
RUN pip install snorkel-0.9.8/
# snorkel breaks the deps, fix em
RUN mamba install -c conda-forge scikit-learn=1.1.1 -y
RUN pip uninstall numpy -y && mamba install -c conda-forge numpy=1.23.4 -y
RUN pip install pandas==1.4.4 && pip install numpy==1.23.4
RUN mamba install -c conda-forge shap=0.41.0 -y

# run/update stage (only update in snorkel will rebuild this stage)
FROM continuumio/miniconda3
COPY --from=build /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:/opt/conda/condabin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SHELL ["mamba", "run", "-n", "viz-inf-cpu", "/bin/bash", "-c"]
# if you move this to end, all updated to code wont rebuild the image
COPY . /visaland-visa-form-utility/
WORKDIR /visaland-visa-form-utility
# install vizard
RUN pip install -e .

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "viz-inf-cpu", \
 "mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
 "--backend-store-uri", "sqlite:///mlflow-inference.db", \
  "--default-artifact-root", "./mlruns-inference/"]
# ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", \
# "viz-inf-cpu" , "python", "-m", "api.main", "--experiment_name", \
# "\"docker-container\"" , "--run_id", "\"0b3700846d8d42f4bb5cf173ec6ee0c3\"", \
# "--bind", "0.0.0.0" , "--gunicorn_port", "8000", "--mlflow_port", \
# "5000", "--verbose", "debug", "--workers", "1"]