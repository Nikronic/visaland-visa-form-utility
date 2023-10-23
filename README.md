# Vizard

## 1 Installation

**tip:** You can use `mamba/micromamba` to hugely speed up the installation process. If you don't want to, replace all instances of the `mamba/micromamba` with `conda` in following steps.

### 1.1 Create a `micromamba` env

Well all packages gonna be here. Note that we have to stick to `python=3.10` as `mlflow` does not support higher versions due to `pyarrow` dependency.
>`micromamba create --name vizard-dev python=3.10 -y`

### 1.2 Activate the new environment

Make sure you activate this environment right away:
>`micromamba activate vizard-dev`

### 1.3 Update `pip`

You should have at least `pip >= 23.1.2`
>`pip install --upgrade pip`

### 1.4 Install PyTorch

At the moment, even though the best model does not need pytorch, but we need to install it as the dependency for AutoML and XAI modules. This is also to make sure AutoML libs (`flaml`) won't install GPU based packages.
>`mamba install pytorch cpuonly -c pytorch -y`

### 1.5 Install `mlflow`

Install the tracking server. This will install a lot of dependencies.
>`mamba install -c conda-forge mlflow==2.7.1 -y`

Now, if you run `bash mlflow-server.sh`, you should see a output like this:

```shell
[2023-10-18 15:16:30 +0330] [19492] [INFO] Starting gunicorn 21.2.0
[2023-10-18 15:16:30 +0330] [19492] [INFO] Listening at: http://0.0.0.0:5000 (19492)
[2023-10-18 15:16:30 +0330] [19492] [INFO] Using worker: sync
[2023-10-18 15:16:30 +0330] [19493] [INFO] Booting worker with pid: 19493
[2023-10-18 15:16:30 +0330] [19494] [INFO] Booting worker with pid: 19494
[2023-10-18 15:16:30 +0330] [19495] [INFO] Booting worker with pid: 19495
[2023-10-18 15:16:30 +0330] [19496] [INFO] Booting worker with pid: 19496
```

### 1.6 Install this package `vizard`

Make sure you are in the root of the project, i.e. the same directory as the repo name. If you are in correct path, you should see `setup.py` containing information about `vizard`.
>`pip install -e .`

After installation, note that dependencies are not installed. Now, we have to install all the dependencies one by one. (`conda` can't do it)

### 1.7 Install data extraction dependencies

>1. `mamba install -c conda-forge xmltodict==0.13.0 -y`
>2. `pip install pikepdf==8.5.1`
>3. `pip install pypdf2==3.0.1`

### 1.8 Install Snorkel weak supervised learner

We need our custom fork of `snorkel` for handling training for unlabeled data (some modules added by me). It relies on PyTorch and that is why we had to install the specific version of PyTorch first.

The custom fork is already included in the directory of the project and can be installed manually:
>`git clone https://github.com/Nikronic/snorkel.git`
>`cd snorkel && pip install -e . && cd ..`

### 1.9 Install ML libs

>1. `mamba install -c conda-forge xgboost==2.0.0 -y`
>2. `mamba install -c conda-forge lightgbm==4.0.0 -y`
>3. `mamba install -c conda-forge catboost==1.2.2 -y`
>4. `mamba install -c conda-forge flaml==2.1.1 -y`
>5. `mamba install -c conda-forge shap==0.43.0 -y`
>6. `mamba install -c conda-forge shapely==2.0.2 -y`

### 1.10 Data Version Control (DVC) tool

Note that for inference we can actually ignore this, but given that we might have tone of data change and monitoring later on, it is best to systematically load data (similar to what MLflow does for runs and Git does for codes)

>1. `mamba install -c conda-forge dvc=2.10.2 -y`
>2. `mamba install -c conda-forge dvclive=0.12.1 -y`

### 1.11 Fix broken dependencies

>1. `mamba install -c conda-forge fsspec=2022.10.0 -y`: required by `dvc`
>2. `mamba install -c conda-forge funcy=1.17 -y`: required by `dvc`

### 1.12 Install API libs

These libraries (the main one is FastAPI) are not for the ML part and only are here to provide the API and web services.

>1. `mamba install -c conda-forge fastapi==0.104.0 -y`
>2. `mamba install -c conda-forge uvicorn==0.23.2 -y`

### 1.13 Install `enlighten` for beauty

This is just a progress bar. It has to be optional but for now, just install it :D.
>`mamba install -c conda-forge enlighten==1.12.0 -y`
