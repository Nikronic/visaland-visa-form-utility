__all__ = [
    "InvitationLetterSenderRelation",
    "INVITATION_LETTER_SENDER_IMPORTANCE",
    "TravelHistoryRegion",
    "TRAVEL_HISTORY_REGION_IMPORTANCE",
]

from enum import Enum
from typing import Dict


class InvitationLetterSenderRelation(Enum):
    CHILD = "child"  # from your child(ren)
    SIBLING = "sibling"  # from your sibling(s)
    PARENT = "parent"  # from your parent(s)
    F2 = "f2"  # from your second-degree family (aunt, uncle, etc)
    F3 = "f3"  # from your third-degree family (children of "f2")
    FRIEND = "friend"  # from your friend (assuming non-legendary person)
    SPOUSE = "spouse"  # from your spouse (you should have had "family" visa already)
    PRO_UNRELATED = "pro_unrelated"  # professional yet not related to your career
    PRO_RELATED = "pro_related"  # professional and aligned with your career
    NONE = "none"  # no invitation letter
    BASE = "base"  # negative effect for not normalizing other features - see #137


INVITATION_LETTER_SENDER_IMPORTANCE: Dict[InvitationLetterSenderRelation, float] = {
    InvitationLetterSenderRelation.CHILD: 1.0,
    InvitationLetterSenderRelation.SIBLING: 0.98,
    InvitationLetterSenderRelation.PARENT: 0.75,
    InvitationLetterSenderRelation.F2: 0.5,
    InvitationLetterSenderRelation.F3: 0.15,
    InvitationLetterSenderRelation.FRIEND: 0.1,
    InvitationLetterSenderRelation.SPOUSE: 0.1,
    InvitationLetterSenderRelation.PRO_UNRELATED: 0.1,
    InvitationLetterSenderRelation.PRO_RELATED: 0.35,
    InvitationLetterSenderRelation.NONE: 0.0,
    InvitationLetterSenderRelation.BASE: -0.45,
}


class TravelHistoryRegion(Enum):
    """Our custom categorization of visa regions

    Note:
        This customization is based on our domain experts and only based on difference in
        acquisition of visa.

    Note:
        There is an exception, where we have included the count for **Schengen** visa.
        The reason is that this is the only region where multiple acquisition of visa has
        different meaning than only having it for once (such as *JP*)

    Note:
        See issue #137 to further know about the reason behind having a negative
        importance for `"base"`.

    Variable description:
     - ``SCHENGEN_ONCE``: Schengen one time
     - ``SCHENGEN_TWICE``: Schengen twice
     - ``US_UK_AU``: United States, United Kingdom and Australia
     - ``JP_KR_AF``: Japan, Korea, Africa
     - ``BR_SG_TH_MY_RU``: Brazil, Singapore, Thailand, Malaysia, Russia
     - ``AE_OM_QA``: Emirates, Oman, Qatar
     - ``AM_GE_TR_AZ``: Armenia, Georgia, Turkiye, Azerbaijan
     - ``NONE``: no travel history

    """

    SCHENGEN_ONCE = "schengen_once"  # Schengen one time
    SCHENGEN_TWICE = "schengen_twice"  # Schengen twice
    US_UK_AU = "us_uk_au"  # United States, United Kingdom and Australia
    JP_KR_AF = "jp_kr_af"  # Japan, Korea, Africa
    BR_SG_TH_MY_RU = "br_sg_th_my_ru"  # Brazil, Singapore, Thailand, Malaysia, Russia
    AE_OM_QA = "ae_om_qa"  # Emirates, Oman, Qatar
    AM_GE_TR_AZ = "am_ge_tr_az"  # Armenia, Georgia, Turkiye, Azerbaijan
    NONE = "none"  # no travel history
    BASE = "base"  # negative effect for not normalizing other features - see #137


TRAVEL_HISTORY_REGION_IMPORTANCE: Dict[TravelHistoryRegion, float] = {
    TravelHistoryRegion.SCHENGEN_ONCE: 0.5,
    TravelHistoryRegion.SCHENGEN_TWICE: 0.75,
    TravelHistoryRegion.US_UK_AU: 0.8,
    TravelHistoryRegion.JP_KR_AF: 0.4,
    TravelHistoryRegion.BR_SG_TH_MY_RU: 0.2,
    TravelHistoryRegion.AE_OM_QA: 0.1,
    TravelHistoryRegion.AM_GE_TR_AZ: 0.05,
    TravelHistoryRegion.NONE: 0.0,
    TravelHistoryRegion.BASE: -0.35
}
"""A mapping from :class:`TravelHistoryRegion` to a percentage based importance

This dictionary used to map user responses to the importance of their response,
and hence, manipulate calculated values such as chance as part of manual insertion
of parameters via :class:`vizard.models.estimators.manual.core.ParameterBuilderBase`

Note:
    The keys of this dictionary are
        :class:`vizard.models.estimators.manual.constant.TravelHistoryRegion`

Note:
    See issue #137 to further know about the reason behind having a negative
    importance for `"base"`.

See Also:
    :class:`vizard.models.estimators.manual.core.TravelHistoryParameterBuilder`
"""
