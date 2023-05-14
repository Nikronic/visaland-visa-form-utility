# Vizard

## 1 Installation

**tip:** You can use `mamba` to hugely speed up the installation process. If you don't want to, replace all instances of the `mamba` with `conda` in following steps.

### 1.1 Create a `conda` env

Well all packages gonna be here.
>`conda create --name viz-inf-cpu python=3.8.0 -y`

### 1.2 Activate the new environment

Make sure you activate this environment right away:
>`conda activate viz-inf-cpu`

### 1.3 Update `pip`

You should have at least `pip >= 23.1.2`
>`pip install --upgrade pip`

### 1.4 Pin the Python version

When using `conda`, `mamba` and so on, it might update the Python to its latest version. We should prevent that by pinning the Python version in the `conda` environment. To do so:

`echo "python 3.8.0" >>$CONDA_PREFIX/conda-meta/pinned`

### 1.5 Install `mlflow`

Install the tracking server. This will install a lot of dependencies.
>`pip install mlflow==1.28.0`

Now, if you run `bash mlflow-inference-server.sh`, you should see a output like this:

```shell
[2023-05-10 15:37:21 +0330] [8394] [INFO] Starting gunicorn 20.1.0
[2023-05-10 15:37:21 +0330] [8394] [INFO] Listening at: http://0.0.0.0:5000 (8394)
[2023-05-10 15:37:21 +0330] [8394] [INFO] Using worker: sync
[2023-05-10 15:37:21 +0330] [8396] [INFO] Booting worker with pid: 8396
[2023-05-10 15:37:21 +0330] [8397] [INFO] Booting worker with pid: 8397
[2023-05-10 15:37:21 +0330] [8398] [INFO] Booting worker with pid: 8398
[2023-05-10 15:37:21 +0330] [8399] [INFO] Booting worker with pid: 8399
```

### 1.6 Install this package `vizard`

Make sure you are in the root of the project, i.e. the same directory as the repo name. If you are in correct path, you should see `setup.py` containing information about `vizard`.
>`pip install -e .`

After installation, note that dependencies are not installed. Now, we have to install all the dependencies one by one. (`conda` can't do it, lol)

### 1.7 Install data extraction dependencies

>1. `mamba install -c conda-forge xmltodict=0.13.0 -y`
>2. `pip install pikepdf==5.1.5`
>3. `pip install pypdf2==2.2.1`

### 1.8 Install PyTorch

At the moment, even though the best model does not need pytorch, but we need to install it as the depenecy for AutoML and XAI modules.
>`mamba install pytorch==1.10.1 cpuonly -c pytorch -y`

### 1.9 Install Snorkel weak supervised learner

We need our custom fork of `snorkel` for handling training for unlabeled data (some modules addes by me). It relies on PyTorch and that is why we had to install the specific version of PyTorch first.

The custom fork is already included in the directory of the project and can be installed manually:
>`pip install snorkel-0.9.8/`

**Note:** This installation messes up a couple of dependencies. But we will fix them manually later on.

### 1.10 Install `enlighten` for beauty

This is just a progress bar. It has to be optional but for now, just install it :D.
>`pip install enlighten==1.10.2`

### 1.11 Install ML libs

>1. `mamba install xgboost==1.3.3 -y`
>2. `mamba install -c conda-forge lightgbm==3.3.3 -y`
>3. `pip install catboost==1.0.6`
>4. `mamba install -c conda-forge flaml=1.0.12 -y`
>5. `mamba install -c conda-forge shap=0.41.0 -y`
>6. `pip install shapely==1.8.5.post1`

### 1.12 Fix broken dependencies

>1. `mamba install -c conda-forge numba=0.56.3 -y`
>2. `mamba install -c conda-forge numpy=1.23.4 -y`

### 1.13 Install API libs

These libraries (the main one is FastAPI) are not for the ML part and only are here to provide the API and web services.

>1. `pip install pydantic==1.9.1`
>2. `pip install fastapi==0.85.0`
>3. `pip install uvicorn==0.18.2`

### 1.14 Data Version Control (DVC) tool

Note that for inference we can actually ignore this, but given that we might have tone of data change and monitoring later on, it is best to systematically load data (similar to what MLflow does for runs and Git does for codes)

>1. `mamba install -c conda-forge dvc=2.10.2 -y`
>2. `mamba install -c conda-forge dvclive=0.12.1 -y`

### 1.15 Again some clueless broken dependencies

>1. `mamba install -c conda-forge fsspec=2022.10.0 -y`
>2. `mamba install -c conda-forge funcy=1.17 -y`

### 1.16 Install `ray` distributed training

>`pip install ray==1.13.0`
