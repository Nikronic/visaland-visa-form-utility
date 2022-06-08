from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

import torch
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
data_labeled = data[(data['VisaResult'] == 'acc') | (data['VisaResult'] == 'rej')]
# Remark: only should be used for evaluation of trained `LabelModel` and no where else
# convert strong to weak temporary so `lf_weak_*` can work
data_labeled['VisaResult'] = data_labeled['VisaResult'].apply(lambda x: 'w-acc' if x == 'acc' else 'w-rej')

# label functions
lfs = [labeling.lf_weak_accept, labeling.lf_weak_reject, labeling.lf_no_idea]

# apply LFs to the unlabeled (for `LabelModel` training) and labeled (for `LabelModel` test)
applier = PandasLFApplier(lfs)
label_matrix_train = applier.apply(data_unlabeled)
# Remark: only should be used for evaluation of trained `LabelModel` and no where else
label_matrix_test = applier.apply(data_labeled)
y_test = data_labeled['VisaResult'].apply(lambda x: labeling.ACC if x == 'w-acc' else labeling.REJ).values
y_train = data_unlabeled['VisaResult'].apply(lambda x: labeling.ACC if x == 'w-acc' else labeling.REJ).values

# LF reports
coverage = (label_matrix_train != labeling.ABSTAIN).mean(axis=0)
for i, f in enumerate(lfs):
    print('{} coverage: {:.1f}%'.format(f.name, coverage[i] * 100))

LFAnalysis(L=label_matrix_train, lfs=lfs).lf_summary()
temp = data_unlabeled.iloc[label_matrix_train[:, 0] == labeling.ACC, :].sample(10, random_state=SEED)
# after viewing the `temp` dataframe, we can see that `ACC` cases make sense, they have high age,
#   >0 children, married, etc.

# train the label model and compute the training labels
label_model = LabelModel(cardinality=2, verbose=True)
label_model.train()
label_model.fit(label_matrix_train, n_epochs=1000, log_freq=100, lr=1e-4,
                optimizer='adam', seed=SEED)

# test the label model
with torch.inference_mode():
    label_model.eval()
    data_unlabeled['AL'] = label_model.predict(L=label_matrix_train, tie_break_policy='abstain')

    # report test accuracy (test data here is our labeled data which is larger (good!))
    metrics = ['accuracy', 'coverage', 'precision', 'recall', 'f1']
    label_model_metrics = label_model.score(L=label_matrix_test, Y=y_test, tie_break_policy='abstain',
                                        metrics=metrics)
    for m in metrics:
        print('Label Model {}: {:.1f}%'.format(m, label_model_metrics[m] * 100))
