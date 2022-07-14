"""Contains preprocessing methods for preparing data solely for estimators in :mod:`vizard.models.estimators <vizard.models.estimators>`

This preprocessors expect "already cleaned" data acquired by :mod:`vizard.data <vizard.data>` 
for sole usage of machine learning models for desired frameworks (let's say
changing dtypes or one hot encoding for torch or sklearn that is only
useful for these frameworks)
"""