"""Hosts all heuristics for integration of models constructed based on domain experts' knowledge

Examples could be if-else models, frequency based statistical models, etc.
"""

# helpers
import logging

# set logger
logger = logging.getLogger(__name__)

from .constant import (INVITATION_LETTER_SENDER_IMPORTANCE,
                       TRAVEL_HISTORY_REGION_IMPORTANCE,
                       InvitationLetterSenderRelation, TravelHistoryRegion)
from .core import (InvitationLetterParameterBuilder,
                   TravelHistoryParameterBuilder)
