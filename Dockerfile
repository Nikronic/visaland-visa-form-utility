# syntax=docker/dockerfile:1

FROM continuumio/miniconda3

RUN conda install -y mamba -c conda-forge

WORKDIR /visaland-visa-form-utility

RUN mamba create --name viz-inf-cpu python=3.8.0 -y

# Make all RUN commands use the conda env
SHELL ["conda", "run", "-n", "viz-inf-cpu", "/bin/bash", "-c"]

# update pip
RUN pip install --upgrade pip

# Pin the Python version
RUN echo "python 3.8.0" >> $CONDA_PREFIX/conda-meta/pinned

# if you move this to end, all updated to code wont rebuild of the image
COPY . /visaland-visa-form-utility/

# install mlflow
RUN pip install mlflow==1.28.0
# install vizard
RUN pip install -e .
# Install data extraction dependencies
RUN mamba install -c conda-forge xmltodict=0.13.0 -y
RUN pip install pikepdf==5.1.5
RUN pip install pypdf2==2.2.1
# Install PyTorch
RUN mamba install pytorch==1.10.1 cpuonly -c pytorch -y
# Install custom fork of Snorkel
RUN pip install snorkel-0.9.8/
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
RUN mamba install -c conda-forge numpy=1.23.4 -y
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

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "viz-inf-cpu", \
 "mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
 "--backend-store-uri", "sqlite:///mlflow-inference.db", \
  "--default-artifact-root", "./mlruns-inference/"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", \
# "viz-inf-cpu" , "python", "-m", "api.main", "--experiment_name", \
# "\"docker-container\"" , "--run_id", "\"cd28a1700e0f4bdf9de6d736e06ca395\"", \
# "--bind", "0.0.0.0" , "--gunicorn_port", "8000", "--mlflow_port", \
# "5000", "--verbose", "debug", "--workers", "1"]