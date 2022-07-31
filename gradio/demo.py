# core
import pandas as pd
import numpy as np
import pickle
# ours
from vizard.models import preprocessors
from vizard.models import trainers
# devops
import dvc.api
import mlflow
# demo
import gradio as gr
# helpers
from typing import List


# data versioning config
PATH = 'raw-dataset/all-dev.pkl'
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.2.3-dev'  # TODO
MLFLOW_RUN_ID = 'd3c45607550e4df093ca3285fcdd0448'
mlflow.set_tracking_uri('http://localhost:5000')

# read data from DVC storage
#   only need this data for columns to build Gradio interface properly
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
data: pd.DataFrame = pd.read_pickle(data_url)
# drop Y column
data = data.drop(columns=['VisaResult'])

# load fitted preprocessing models
X_CT_NAME = 'train_sklearn_column_transfer.pkl'
x_ct_path = mlflow.artifacts.download_artifacts(run_id=MLFLOW_RUN_ID,
                                                artifact_path=f'models/{X_CT_NAME}',
                                                dst_path=f'gradio/')
with open(x_ct_path, 'rb') as f:
    x_ct: preprocessors.ColumnTransformer = pickle.load(f)

# load fitted FLAML AutoML model for prediction
FLAML_AUTOML_NAME = 'flaml_automl.pkl'
flaml_automl_path = mlflow.artifacts.download_artifacts(run_id=MLFLOW_RUN_ID,
                                                        artifact_path=f'models/{FLAML_AUTOML_NAME}',
                                                        dst_path=f'gradio/')
with open(flaml_automl_path, 'rb') as f:
    flaml_automl: trainers.AutoML = pickle.load(f)

def predict(*args):
    # convert to dataframe
    x_test = pd.DataFrame(data=[list(args)], columns=data.columns)
    x_test = x_test.astype(data.dtypes)
    x_test = x_test.to_numpy()
    # preprocess test data
    xt_test = x_ct.transform(x_test)
    # predict
    y_pred = flaml_automl.predict_proba(xt_test)
    y_pred = y_pred[0, np.argmax(y_pred)]
    msg = f'Probability: {y_pred:.2f}'
    print(msg)
    return msg, y_pred

# build gradio interface given dataframe dtype
inputs: List[gr.components.Component] = []
for col in data.columns:
    if data[col].dtype == 'category':
        inputs.append(gr.Dropdown(choices=list(data[col].unique()),
                                  label=col))
    elif data[col].dtype == 'float32':
        inputs.append(gr.Number(label=col))
    elif data[col].dtype == 'int32':
        inputs.append(gr.Slider(minimum=int(data[col].min()),
                                maximum=int(data[col].max()),
                                step=1,
                                label=col))
    else:
        raise ValueError(f'Unknown dtype {data[col].dtype} for column {col}')

# prediction probability output
outputs = [
    gr.Textbox(label='Probability of acceptance'),
    gr.Number(label='Probability'),
]

examples = [
    [True,'Male',1.1,1,0,0,True,'UAE',3,False,'tourism',6000,'hotel','hotel',True,'bachelor',1.1,'employee',0,'OTHER',0,'OTHER',0,False,False,False,24,0,0,30,0,0,4,15,4,2,0,0,7,5,5,9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',0,42,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'Probability: 1.00',0.9999999539546803,'','','2022-07-31 15:37:40.655966'],
    [True,'Male',4.1,1,0,0,True,'UAE',3,False,'family visit',20000,'f1','f1',True,'phd',1.1,'specialist',1.1,'OTHER',0,'OTHER',0,False,False,True,40,0,0,30,10,0,4,15,2,15,0,0,5,5,5,7,'son',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',37,60,65,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,'Probability: 0.81',0.8066482879561396,'','','2022-07-31 15:45:49.482404'],
    [False,'Male',1.1,1,1,1,True,'TURKEY',6,False,'tourism',20000,'hotel','hotel',True,'phd',1.1,'specialist',1.1,'OTHER',1,'OTHER',1,False,False,True,40,0,0,30,10,0,4,15,2,15,0,0,5,5,5,7,'son',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',9,'other',37,60,65,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,'Probability: 0.95',0.9533567493228952,'','','2022-07-31 15:47:35.890893'],
]

title = 'Vizard'
description = 'Vizard is a tool for predicting the chance of obtaining a visa.'

# launch gradio
gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    flagging_dir='gradio/flags',
    analytics_enabled=True,
).launch(debug=True, enable_queue=True, server_port=7860)
