# ours
from vizard.utils.helpers import LoggerWriter
# helpers
import logging
import sys

# set logger
logger = logging.getLogger(__name__)

# To access the original stdout/stderr, use sys.__stdout__/sys.__stderr__
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)
