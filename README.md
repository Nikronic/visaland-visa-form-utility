# Vizard

## 1 Installation

### 1.1 Docker

Steps:

1. Build the image
2. Run the image
3. Run inference API

#### 1.1.1 Build the image

Simply, build the image using given `Dockerfile-dev`:

```bash
docker build --progress=plain -t vizard:latest -f Dockerfile-dev .
```

##### For people located in Iran

Since Docker Hub is part of the US sanctions, you must resolve it first. For this, you have two options:

1. Getting a server outside of Iran: this is similar to setting up a VPN. Note that in this case, the IP is most likely part of the IR's censorship. So if you plan to deploy your service, you will need a VPN (or any other method that breaks the filtering)
2. Getting a server in Iran: In this case, your don't need a VPN to access to IP/Port. Yet, since sanctions are existing in this case too, you need to change DNS or setup a VPN on your server. For doing that, you can use the DNS provided on [shecan.ir](https://shecan.ir) and apply it on your server using this guide on [configuring DNS on Ubuntu](https://askubuntu.com/a/1392751/1112620)

#### 1.1.2 Run the image

```bash
docker run -it -p 5000:5000 -p 8000:8000 vizard
```

The output should look something like this which indicated `mlflow` tracking server is running:

```bash
[2023-10-24 09:16:55 +0000] [22] [INFO] Starting gunicorn 21.2.0
[2023-10-24 09:16:55 +0000] [22] [INFO] Listening at: http://0.0.0.0:5000 (22)
[2023-10-24 09:16:55 +0000] [22] [INFO] Using worker: sync
[2023-10-24 09:16:55 +0000] [23] [INFO] Booting worker with pid: 23
[2023-10-24 09:16:55 +0000] [24] [INFO] Booting worker with pid: 24
[2023-10-24 09:16:55 +0000] [25] [INFO] Booting worker with pid: 25
[2023-10-24 09:16:55 +0000] [26] [INFO] Booting worker with pid: 26
```

Note that this image uses port 5000 for `mlflow` tracking server and 8000 for `fastapi` inference server.

#### 1.1.3 Run inference API

To run the inference server which is based on `fastapi`, we run a command inside the currently running container.

For instance, if the command from step `1.1.2` succeeded, then you can use `docker ps` to see the running containers. E.g.,

```bash
docker ps
```

The output should look something like this:

```bash
CONTAINER ID   IMAGE           COMMAND                  CREATED         STATUS         PORTS                                                                                  NAMES
e5175efd685b   vizard:0.20.0   "mamba run --no-capt…"   3 minutes ago   Up 3 minutes   0.0.0.0:5000->5000/tcp, :::5000->5000/tcp, 0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   thirsty_wright
```

Now, run the inference server given the container name from `docker ps`:

```bash
docker exec -it thirsty_wright mamba run --no-capture-output -n viz-inf-cpu python -m api.main --experiment_name "YOUR_DESIRED_NAME" --run_id "THE_RUN_ID_ASSOCIATED_WITH_A_REGISTERED_MLFLOW_MODEL" --bind 0.0.0.0 --gunicorn_port 8000 --mlflow_port 5000 --verbose debug --workers 1
```

Here, `--gunicorn_port` refers to `fastapi` API port.

To find `run_id`, open the `mlflow` service which is on [localhost:5000](http://127.0.0.1:5000), and select **Models** tab, and choose the model you like e.g., `v0.20.0-d2.0.1`. There you need to select the version of the model you want. After this, you will see a field called **Source Run**, e.g., `amusing-ox-207`. After this step, you will find the run id (here, associated with `amusing-ox-207`) which is `3c7ad5db48bf48d9957087231e870ac0`. This is the id associated with the trained model and when running the inference server, this is how you can use your desired model.

Now, you can open [localhost:8000/docs](http://127.0.0.1:8000/docs) and view the OpenAPI documentation and test the APIs.

### 1.2 Mamba

**tip:** You can use `mamba/micromamba` to hugely speed up the installation process. If you don't want to, replace all instances of the `mamba/micromamba` with `conda` in following steps.

#### 1.2.1 Create a `micromamba` env

Well all packages gonna be here. Note that we have to stick to `python=3.10` as `mlflow` does not support higher versions due to `pyarrow` dependency.

```bash
micromamba create --name vizard-dev python=3.10 -y
```


#### 1.2.2 Activate the new environment

Make sure you activate this environment right away:

```bash
micromamba activate vizard-dev
```

#### 1.2.3 Update `pip`

You should have at least `pip >= 23.1.2`. To update to the latest version:

```bash
pip install --upgrade pip
```

#### 1.2.4 Install PyTorch

At the moment, even though the best model does not need pytorch, but we need to install it as the dependency for AutoML and XAI modules. This is also to make sure AutoML libs (`flaml`) won't install GPU based packages.

```bash
mamba install pytorch==2.1.0 cpuonly -c pytorch -y
```

#### 1.2.5 Install `mlflow`

Install the tracking server. This will install a lot of dependencies which we need directly, but any updated version which will be installed by `mlflow` would be enough.

```bash
mamba install -c conda-forge mlflow==2.8.0 -y
```

Now, if you run `bash mlflow-server.sh`, you should see a output like this:

```bash
[2023-10-18 15:16:30 +0330] [19492] [INFO] Starting gunicorn 21.2.0
[2023-10-18 15:16:30 +0330] [19492] [INFO] Listening at: http://0.0.0.0:5000 (19492)
[2023-10-18 15:16:30 +0330] [19492] [INFO] Using worker: sync
[2023-10-18 15:16:30 +0330] [19493] [INFO] Booting worker with pid: 19493
[2023-10-18 15:16:30 +0330] [19494] [INFO] Booting worker with pid: 19494
[2023-10-18 15:16:30 +0330] [19495] [INFO] Booting worker with pid: 19495
[2023-10-18 15:16:30 +0330] [19496] [INFO] Booting worker with pid: 19496
```

#### 1.2.6 Install this package `vizard`

Make sure you are in the root of the project, i.e. the same directory as the repo name. If you are in correct path, you should see `setup.py` containing information about `vizard`.

```bash
pip install -e .
```

After installation, note that dependencies are not installed. Now, we have to install all the dependencies one by one. (`conda` can't do it)

#### 1.2.7 Install data extraction dependencies

```bash
mamba install -c conda-forge xmltodict==0.13.0 -y
pip install pikepdf==8.5.1
pip install pypdf==3.1.0
```

#### 1.2.8 Install Snorkel weak supervised learner

We need our custom fork of `snorkel` for handling training for unlabeled data (some modules added by me). It relies on PyTorch and that is why we had to install the specific version of PyTorch first.

The custom fork is already included as a git submodule and can be obtained by running the following command in the root of this repo:

```bash
git submodule update --init --recursive
cd snorkel
pip install -e .
cd ..
```

#### 1.2.9 Install ML libs

These all are related to ML:

```bash
mamba install -c conda-forge xgboost==2.0.0 -y
mamba install -c conda-forge lightgbm==4.0.0 -y
mamba install -c conda-forge catboost==1.2.2 -y
mamba install -c conda-forge flaml==2.1.1 -y
mamba install -c conda-forge shap==0.43.0 -y
mamba install -c conda-forge shapely==2.0.2 -y
```

#### 1.2.10 Data Version Control (DVC) tool

Note that for inference we can actually ignore this, but given that we might have tone of data change and monitoring later on, it is best to systematically load data (similar to what MLflow does for runs and Git does for codes)

```bash
mamba install -c conda-forge dvc=2.10.2 -y
mamba install -c conda-forge dvclive=0.12.1 -y
```

#### 1.2.11 Fix broken dependencies

```bash
mamba install -c conda-forge fsspec=2022.10.0 -y
mamba install -c conda-forge funcy=1.17 -y
```

Note that `fsspec=2022.10.0` and `funcy=1.17` required by `dvc=2.10.2`.

#### 1.2.12 Install API libs

These libraries (the main one is FastAPI) are not for the ML part and only are here to provide the API and web services.

```bash
mamba install -c conda-forge fastapi==0.104.0 -y
mamba install -c conda-forge uvicorn==0.23.2 -y
```

#### 1.2.13 Install `enlighten` for beauty

This is just a progress bar. It has to be optional but for now, just install it :D.

```bash
mamba install -c conda-forge enlighten==1.12.0 -y
```

## For developers

### Testing

Since we are using `FastAPI` and there is an elegant way for testing with it, we install a few more dependencies (of course `pytest` as our go-to for reasons I don't know!):

```bash
mamba install -c conda-forge httpx=0.25.1 -y
mamba install -c anaconda pytest=7.4.0
```

*Note:* you need to make sure MLflow server is running. This tests only can handle FastAPI APIs without explicit running instance. (i.e., run a MLflow server)

### Mlflow migration of models from training to production

#### Description

Since the training is done in local machine, when I exported and then imported these trained models into `mlflow-inference.db`, all links had references to my local machine path i.e., `/home/nik/....`. I removed all these links manually by editting the database file. Note that, the directory of `mlruns-inference` is fine and requires no modification (unlike previous cases).

This is done thanks to:

1. [mlflow import export utility](https://github.com/mlflow/mlflow-export-import/blob/master/README_single.md): enables exporting registered models from one server and then importing them into another mlflow db and server.
2. [DB Browser for SQLite](https://sqlitebrowser.org/): enables modifying the mlflow db manually to remove all those absolute paths of local machine

#### Steps

Steps needed to migrate a registered model trained on server A (e.g., local machine) to production/serve server B (e.g., cloud, docker):

1. Start mlflow on server A

```bash
# run mlflow of SERVER A on port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000
```

2. Export mlflow registered model:

```bash
export-model --model v0.20.0-d2.0.1 --output-dir temp_ --stages Staging
```

3. turn on the mlflow on server B (I assume it is on port 5000 too).

4. Import exported model from step `2`:

```bash
import-model --model v0.20.0-d2.0.1 --experiment-name train_on_v0.20.0-d2.0.1-HOTFIX \--input-dir temp_
```

5. Open the mlflow database (e.g., `mlflow-inference.db`) on server B and manually edit entries that include absolute path of server A with their relative path. For that, tables `model_versions` and `runs`.
