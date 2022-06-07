from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
import pandas as pd

# our packages
from vizard_snorkel import labeling

# globals
SEED = 322

# load dataframe
path = 'raw-dataset/all-dev.pkl'
data = pd.read_pickle(path)

# exclude labeled data (only include weak labels and no labels)

data_unlabeled = data[(data['VisaResult'] != 'acc') & (data['VisaResult'] != 'rej')]

# label functions
lfs = [labeling.lf_weak_accept, labeling.lf_weak_reject, labeling.lf_no_idea]

# apply the LFs to the unlabeled training data
applier = PandasLFApplier(lfs)
label_matrix_data = applier.apply(data_unlabeled)

# train the label model and compute the training labels
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(label_matrix_data, n_epochs=500, log_freq=50, seed=SEED)
data_unlabeled['AL'] = label_model.predict(L=label_matrix_data, tie_break_policy='abstain')
z = pd.concat([data_unlabeled['AL'], data_unlabeled['VisaResult']], axis=1)
print
