#!/bin/bash

mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow-inference.db --default-artifact-root ./mlruns-inference/
