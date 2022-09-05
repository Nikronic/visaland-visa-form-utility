# core
from dtreeviz import trees
from dtreeviz.models.xgb_decision_tree import ShadowXGBDTree
import numpy as np
import pandas as pd
import flaml

# helpers
import logging


logger = logging.getLogger(__name__)

flaml_model: flaml.AutoML
xt_train: np.ndarray
yt_train: np.ndarray
data_train: pd.DataFrame
data_test: pd.DataFrame
target: str = 'VisaResult'

xgb_shadow = ShadowXGBDTree(
    flaml_model.model.estimator,
    1,
    xt_train,
    yt_train,
    data_train.columns,
    data_test.columns,
    class_names=list(data_train.loc[0, target].cat.categories.values)
)

trees.viz_leaf_samples(xgb_shadow)
trees.dtreeviz(xgb_shadow)
trees.describe_node_sample(xgb_shadow, 1)
print(trees.explain_prediction_path(
    xgb_shadow,
    xt_train[0],
    explanation_type='plain_english')
)
# and so on

# https://github.com/parrt/dtreeviz/blob/master/notebooks/dtreeviz_xgboost_visualisations.ipynb
