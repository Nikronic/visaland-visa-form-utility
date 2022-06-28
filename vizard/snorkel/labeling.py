__all__ = [
    'labeling_function', 'lf_weak_reject', 'lf_no_idea', 'ABSTAIN', 'REJ', 'ACC'
]

# core
import pandas as pd
# snorkel
from snorkel.labeling import labeling_function


# TODO: move this to `vizard.snorkel.constant.py`
# define the label mappings
ABSTAIN = -1
REJ = 0  # TODO: cant be 2 so it matches our dataframe already?
ACC = 1



# convert weak accept to accept (=='w-acc' -> 'acc')
@labeling_function()
def lf_weak_accept(x: pd.Series) -> int:
    if x['VisaResult'] == 'w-acc':    # 3 == weak acc
        return ACC
    else:
        return ABSTAIN

# convert weak reject to reject (=='w-rej' -> 'rej')
@labeling_function()
def lf_weak_reject(x: pd.Series) -> int:
    if x['VisaResult'] == 'w-rej':  # 4 == weak rej
        return REJ
    else:
        return ABSTAIN

# convert no idea to reject (=='no idea' -> 'rej')
@labeling_function()
def lf_no_idea(x: pd.Series) -> int:
    """
    Ms. S's suggestion was that if she can't remember, then it's probably a `rej` (=rejected) case
    """
    if x['VisaResult'] == 'no idea':  # 5 == no idea
        return REJ
    else:
        return ABSTAIN
