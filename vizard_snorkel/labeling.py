from snorkel.labeling import labeling_function
import pandas as pd


# define the label mappings
ABSTAIN = -1
REJ = 0  # TODO: cant be 1 so it matches our dataframe already?
ACC = 1



# convert weak accept to accept (=='w-acc' -> 'acc')
@labeling_function()
def lf_weak_accept(x: pd.Series) -> int:
    # only unlabeled data is allowed here
    assert (x['VisaResult'] != 'acc') | (x['VisaResult'] != 'rej')
    if x['VisaResult'] == 'w-acc':    # 3 == weak acc
        return ACC
    else:
        return ABSTAIN

# convert weak reject to reject (=='w-rej' -> 'rej')
@labeling_function()
def lf_weak_reject(x: pd.Series) -> int:
    # only unlabeled data is allowed here
    assert (x['VisaResult'] != 'acc') | (x['VisaResult'] != 'rej')
    if x['VisaResult'] == 'w-rej':  # 4 == weak rej
        return REJ
    else:
        return ABSTAIN

# convert no idea to reject (=='no idea' -> 'rej')
@labeling_function()
def lf_no_idea(x: pd.Series) -> int:
    # only unlabeled data is allowed here
    assert (x['VisaResult'] != 'acc') | (x['VisaResult'] != 'rej')
    if x['VisaResult'] == 'no idea':  # 5 == no idea
        return REJ
    else:
        return ABSTAIN
