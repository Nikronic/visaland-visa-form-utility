__all__ = [
    'InvitationLetterSenderRelation', 'INVITATION_LETTER_SENDER_IMPORTANCE'
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
}
