"""Contains configs of training and scripts for it

Configs contain different setups for training that need to used by different
training scripts defined in this module. 
Also, for each different type of estimator or different purpose, a separate
training script needs to be defined.

Note that training scripts must be complete. I.e. they should be able to
checkpoint, load model, load data, etc.
"""

# helpers
import logging


# set logger
logger = logging.getLogger(__name__)
