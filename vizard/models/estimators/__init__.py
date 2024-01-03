"""Contains any model/estimator that can be used for predicting or manipulating values.

It could be as simple as an API for weighted average, a machine learning
model such as *SVM* or *RandomForest* to super complicated
neural network models such as *ResNet*. The name `estimators` is inspired
by sklearn-estimator_.

Submodules:

    * :mod:`vizard.models.estimators.manual`: Any method that enables integration of variables, 
        parameters, etc that are provided manually. 


.. _sklearn-estimator: https://scikit-learn.org/stable/glossary.html#term-estimator
"""

# helpers
import logging

# set logger
logger = logging.getLogger(__name__)


