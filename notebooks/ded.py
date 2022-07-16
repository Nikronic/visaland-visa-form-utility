#>
# core 3rd
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ours: data
from vizard.data import functional
from vizard.data import constant
from vizard.data import logic
from vizard.data import preprocessor
# ours: utils
from vizard.utils.visualization import add_percentage_axes
# ours: configs/db
from vizard.configs import CANADA_COUNTRY_CODE_TO_NAME

# MLOps
from mlflow.tracking import MlflowClient
import mlflow
import dvc.api

# utils
from IPython.display import display
from typing import Any
import logging
import shutil
import sys
import os

'''
## Reproducibility

'''

'''
### Setup DVC experiment and configs
'''

#>
# main path
SRC_DIR = '/mnt/e/dataset/processed/all/'  # path to source encrypted pdf
DST_DIR = 'raw-dataset/all/'  # path to decrypted pdf

# DVC: main dataset
PATH = DST_DIR[:-1] + '.pkl'  # path to source data, e.g. data.pkl file
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.0.3'  # use latest using `git tag`

# DVC: helper - (for more info see the API that uses these files)
# data file for converting country names to continuous score in "economical" sense
HELPER_PATH_GDP = 'raw-dataset/API_NY.GDP.PCAP.CD_DS2_en_xml_v2_4004943.pkl'
HELPER_VERSION_GDP = 'v0.1.0-field-GDP'  # use latest using `git tag`
# data file for converting country names to continuous score in "all" possible senses
HELPER_PATH_OVERALL = 'raw-dataset/databank-2015-2019.pkl'
HELPER_VERSION_OVERALL = 'v0.1.0-field'  # use latest using `git tag`
# gather these for MLFlow track
all_helper_data_info = {
    HELPER_PATH_GDP: HELPER_VERSION_GDP,
    HELPER_PATH_OVERALL: HELPER_VERSION_OVERALL,
}

'''
### Configure Logging
'''

#py>
logger = logging.getLogger(__name__)  # subset of `vizard_ipynb` logger
# override display to log using logger
display = logger.info
# if the script is run directly, log to console instead of inheriting module level log
if __name__ == '__main__':
    logger_handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(logger_handler)
    logger.setLevel(logging.INFO)

#nb>
#VERBOSITY = logging.INFO
#logger = logging.getLogger(__name__)
#logger.setLevel(VERBOSITY)
#
## Set up root logger, and add a file handler to root logger
#if not os.path.exists(REPO + '/artifacts'):
#    os.makedirs(REPO + '/artifacts')
#    os.makedirs(REPO + '/artifacts/logs')
#    os.makedirs(REPO + '/artifacts/notebooks')
#
#logger_handler = logging.FileHandler(filename=REPO + '/artifacts/logs/nb.log', mode='w')
#logger.addHandler(logger_handler)

#>
SEED = 322
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

#nb>
#logger.info('SEED={}'.format(SEED))

'''
## Loading Data

'''

'''
### Setup MLFlow
'''

#nb>
## NBVAL_IGNORE_OUTPUT
## log experiment configs
#MLFLOW_EXPERIMENT_NAME = 'EDA resulted data manipulation via automated nb2py'
#mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
#MLFLOW_TAGS = {
#    'stage': 'dev'  # dev, beta, production
#}
#mlflow.set_tags(MLFLOW_TAGS)
#client = MlflowClient()
#
#logger.info('MLFlow experiment name: {}'.format(MLFLOW_EXPERIMENT_NAME))
#logger.info('MLFlow experiment id: {}'.format(mlflow.active_run().info.run_id))
#logger.info('DVC data version: {}'.format(VERSION))
#logger.info('DVC repo (root): {}'.format(REPO))
#logger.info('DVC data source path: {}'.format(PATH))

'''
### Set global options for the notebook
'''

#nb>
## pandas
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 50)
#
## matplotlib
#SMALL_SIZE = 18
#MEDIUM_SIZE = 22
#BIGGER_SIZE = 26
#
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#
#plt.style.use('fivethirtyeight')
#import mplcyberpunk
#plt.style.use('cyberpunk')
#%matplotlib inline

'''
### Import data from DVC storage
'''

#>
# get url data from DVC data storage
data_url = dvc.api.get_url(path=PATH, repo=REPO, rev=VERSION)
# read dataset from remote (local) data storage
data = pd.read_pickle(data_url)

'''
### Global Data Preprocessing Functions
'''

#>
canada_logic = logic.CanadaLogics()
# just to make sure mypy does not complain after `vizard_ipynb.nb2py.py conversion`
feature_name: Any = ''

'''
### Getting to know the data
'''

#nb>
#display(data.sample(5, random_state=SEED))

#>
# remove duplicates
data.drop_duplicates(inplace=True)

#nb>
#display(data.isnull().sum()[data.isnull().sum() != 0])  # checking for total null values

'''
it seems we have 4 rows with no information (empty forms) but the flag exists. There is nothing we can do except deleting them.
'''

#nb>
#feature_name = 'P1.PD.VisaType.VisaType'
#data[data[feature_name].isna()]

#>
feature_name = 'P1.PD.VisaType.VisaType'
data = data.drop(data[data[feature_name].isna()].index)
display(data.isnull().sum()[data.isnull().sum() != 0])

'''
We can see the only three type of data are missing:
1. COB: country of birth
2. Period: age of sibling and kids (probably because the person does not have any kid or sibling to fill)
3. Address: Address of applicants, their spouses, parents, children, and siblings. Although full address is available, only city matters.

| Feature | Fill | Desc |
| --- | --- | --- |
| COB | str: same as applicant's COB for ghost spouse | If all SpsCOB are the same, just drop the entire column, so no differentiation between ghost spouse or missed one.  
| Period | int: 0 | prob only ghost cases, otherwise, should be filled statistically 
| Address | str: city name | If None (for both ghost case and missing info), skip it (None). Since the value itself is not important and we prefer the difference of applicant's value from their family (to count long distance cases), we can only focus on filled values that are different from applicant's.
'''

#>
feature_name = [c for c in data.columns.values if 'Sps' in c]
display(feature_name)

#nb>
#data[data[feature_name[0]].isna()][feature_name]

#nb>
#display(data[data[feature_name[0]].isna()][feature_name].__len__())
#display(data[(data[feature_name[2]] == 'OTHER') & (data[feature_name[0]].isna())][feature_name].__len__())

'''
There are 70-67 cases where the person has job title but no COB, which means that only 3 cases are not ghost cases and we can fill their COB using their spouse's COB.
For all other cases (ghosts), we just use the dominant case, IRAN. 
'''

#>
# fill ghost cases' COB with 'IRAN'
data.loc[(data[feature_name[2]] == 'OTHER') & (data[feature_name[0]].isna()), [feature_name[0]]] = 'IRAN'

#nb>
#display(data[data[feature_name[0]].isna()][feature_name].__len__())
#display(data[(data[feature_name[2]] == 'OTHER') & (data[feature_name[0]].isna())][feature_name].__len__())

#>
# fill non-filled spouse COB cases' with their spouses' COB
data.loc[data[feature_name[0]].isna(), [feature_name[0]]] = data['P1.PD.PlaceBirthCountry']

#nb>
#display(data[data[feature_name[0]].isna()][feature_name].__len__())
#display(data[(data[feature_name[2]] == 'OTHER') & (data[feature_name[0]].isna())][feature_name].__len__())

'''
## Analyzing the features
'''

'''
### Ground truth
'''

#>
output_name = 'VisaResult'
data.loc[data[output_name] == 1, [output_name]] = 'acc'  # 1 -> accepted
data.loc[data[output_name] == 2, [output_name]] = 'rej'  # 2 -> rejected
data.loc[data[output_name] == 3, [output_name]] = 'w-acc'  # 3 -> allegedly accepted
data.loc[data[output_name] == 4, [output_name]] = 'w-rej'  # 4 -> allegedly rejected
data.loc[data[output_name] == 5, [output_name]] = 'no idea'  # 5 -> no idea
output_hue_order = ['acc', 'rej', 'w-rej', 'w-acc', 'no idea']

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[output_name].value_counts().plot.pie(
#    autopct='%1.2f%%', ax=ax[0], shadow=True, textprops={'color': 'black'})
#ax[0].set_title(output_name)
#ax[0].set_ylabel('')
#sns.countplot(x=output_name, data=data, ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title(output_name)
#plt.show()

'''
We don't have the labels for %18 of our data, but we have weak labels for around %10. In this way, we can revive 80% of labels by doing weak supervised learning. Check [Snorkel](https://www.snorkel.org/get-started/) for more information. 
'''

'''
### P1.AdultFlag -> categorical
'''

#>
feature_name = 'P1.AdultFlag'
output_name = 'VisaResult'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
data[data[feature_name] == False]

'''
Ok, useless kids, dropping all of them, and hence, the entire `P1.AdultFlag` flag
'''

#>
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.ServiceIn.ServiceIn -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.ServiceIn.ServiceIn' 
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.ServiceIn.ServiceIn -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.VisaType.VisaType' 
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data, ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()

'''
TODO: They seem proportional, for now, we skip it, but if after modeling, we didn't find any improvement by including this feature, we must remove it.
'''

'''
### P1.PD.Sex.Sex -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.Sex.Sex', 'P1.MS.SecA.MS'
display(data.groupby([feature_name[0], output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(2, 2, figsize=(22, 14))
#data[[feature_name[0], output_name]].groupby(
#    [feature_name[0]]).count().plot.bar(ax=ax[0, 0])
#add_percentage_axes(ax[0, 0], len(data))
#ax[0, 1].set_title('{} vs {}'.format(output_name, feature_name[0]))
#sns.countplot(x=feature_name[0], hue=output_name, data=data,
#    ax=ax[0, 1], hue_order=output_hue_order)
#add_percentage_axes(ax[0, 1], len(data))
#ax[0, 1].set_title('{}: {}'.format(feature_name[0], output_name))
#
#z = data[data[feature_name[1]] == '02']  # singles data ('02' == single)
#z[[feature_name[0], output_name]].groupby(
#    [feature_name[0]]).count().plot.bar(ax=ax[1, 0])
#add_percentage_axes(ax[1, 0], len(z))
#ax[1, 0].set_title('{} vs {}'.format(output_name, feature_name[0]))
#sns.countplot(x=feature_name[0], hue=output_name, data=z,
#    ax=ax[1, 1], hue_order=output_hue_order)
#add_percentage_axes(ax[1, 1], len(z))
#ax[1, 1].set_title('{}: {}'.format(feature_name[0], output_name))
#
#plt.show()

'''
Insights: 
1. It seems that women are getting more visas than man, even though the number of applicants in term of gender are almost equal.
2. A closer looks shows that in **singles**, men are doing better.
'''

'''
### P1.PD.PlaceBirthCity -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PlaceBirthCity'

#nb>
## NBVAL_IGNORE_OUTPUT
#pd.crosstab(data[feature_name], data[output_name],
#            margins=True).style.background_gradient(cmap='summer_r')

'''
The data fro other cities except Tehran, Karaj and Shiraz is too small to matter. At the end of the day, the value of land is summarized into assets, so no point in this, hence deleted.
'''

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.PlaceBirthCountry -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PlaceBirthCountry'
display(data.groupby([feature_name, output_name])[output_name].count())

'''
Well, all are from damned Iran, so let's drop it.
'''

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.Citizenship.Citizenship -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.Citizenship.Citizenship'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.CurrCOR.Row2.Country -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CurrCOR.Row2.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

'''
511 is Canada and someone who has residency of Canada absolutely needs to be excluded as an anomaly.
'''

#>
data.drop(data[data[feature_name] == '511'].index, inplace=True)

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME

data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP))
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=eco_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[feature_name].unique())

'''
It is hard to say that how having different country of residence than IRAN effects the output variable. So know more about it, especially if it is **anomaly** or not, we currently skip it and after having a clean data, we use visualization, statistical or ML methods to figure this out. [See this for more information](https://www.analyticsvidhya.com/blog/2021/04/dealing-with-anomalies-in-the-data/)

We can also make this into categorical, [`IRAN`, `OTHER`] and simplify things, but if it's anomaly, I prefer fixing it.

TODO: last priority but mandatory
'''

'''
### P1.PD.CurrCOR.Row2.Status -> categorical
'''

#>
feature_name = ['P1.PD.CurrCOR.Row2.Country', 'P1.PD.CurrCOR.Row2.Status'] 
data[data[feature_name[0]] != 'IRAN'][feature_name[1]]

'''
Similar as [P1.PD.CurrCOR.Row2.Country](#P1.PD.CurrCOR.Row2.Country)
'''

'''
### P1.PD.CurrCOR.Row2.Other -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CurrCOR.Row2.Other'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.PCRIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PCRIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data, ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()

'''
It is hard to say, but having previous country of residence has higher acceptance rate than none ones. Well, this is maybe the case, because they showed that they have resided in third-party countries and went back to their home country.
'''

'''
### P1.PD.PrevCOR.Row[i].\* -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PrevCOR.Row2.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

'''
Ok, we have good amount of examples that cannot be ignored or labeled as anomaly.

The main issue here is that the countries are so diverse that cannot be considered. To fix this, we consider label feature `P1.PD.PCRIndicator` which is a binary variable indicating if the candidate has had previous country of residence or not. The below cell confirms this, so we drop this column and second row of the same info, i.e. `P1.PD.PrevCOR.Row2.Country`, `P1.PD.PrevCOR.Row2.Status`, `P1.PD.PrevCOR.Row3.Country`, `P1.PD.PrevCOR.Row3.Status`.

All these columns can be integrated into another column, aggregating all of these into sum of non-`OTHER` status for PrevCOR, simply put, the count of previous countries of residence. In this case, `0` means no PCR, so we can get rid of `P1.PD.PCRIndicator` too.
'''

#>
display(data[data['P1.PD.PCRIndicator'] == True].groupby([feature_name, output_name])[output_name].count())

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PrevCOR.Row2.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
output_name = 'VisaResult'
display(data.groupby(['P1.PD.PrevCOR.Row2.Status', 'P1.PD.PrevCOR.Row3.Status']).size())

#>
r = re.compile('P1.PD.PrevCOR.Row..Period')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])

# replace rows of previous country of residency with count of them
agg_column_name = 'P1.PD.PrevCOR.Row.Count'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_previous_residency_country,
                                   agg_column_name=agg_column_name, columns=feature_name)

# delete redundant columns tnx to newly created `agg_column_name='P1.PD.PrevCOR.Row.Count'`
data.drop(['P1.PD.PrevCOR.Row2.Status', 'P1.PD.PrevCOR.Row3.Status', 'P1.PD.PCRIndicator'], axis=1, inplace=True)

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PrevCOR.Row2.Period', 'P1.PD.PrevCOR.Row3.Period', 'P1.PD.PrevCOR.Row.Count'
display(data.groupby([*feature_name, output_name])[output_name].count())

#nb>
#data.sample(5, random_state=SEED)

'''
#### P1.PD.PrevCOR.Row[i].Country -> categorical -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PrevCOR.Row2.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME

data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP))
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=eco_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[feature_name].unique())

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.PrevCOR.Row3.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME
data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP))
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=eco_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[feature_name].unique())

'''
### P1.PD.SameAsCORIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.SameAsCORIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()
#

'''
It seems that having residency of other countries improves the chance of getting the visa. Seems like that the person had a choice to be the citizen of any country (by being refugee of course) but legally has chosen another country than the visiting country (here Canada). I.e. the person has no intention to go back to Iran, but no intention to stay in Canada too, a third-party country is taking care of it!
'''

'''
### P1.PD.CWA.Row2.Country -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = ['P1.PD.CWA.Row2.Country', 'P1.PD.PrevCOR.Row.Count']
display(data.groupby([*feature_name, output_name])[output_name].count())

#>
# convert numbers to countries and transform them into categories that are more frequent
number_to_country = {
    '049': 'Armenia',
    '052': 'Georgia',
    '280': 'UAE',
    'IRAN': 'TURKEY',  # CWA cannot be Iran, prob Turkey
    'OTHER': 'TURKEY',  # CWA cannot be OTHER, prob Turkey
    'TURKEY': 'TURKEY',
}

def fix_cwa(string: str, dic):
    if string in dic.keys():
        string = string.replace(string, dic[string])
    else:
        string = 'OTHER'
    return string

feature_name = 'P1.PD.CWA.Row2.Country'
data[feature_name] = data[feature_name].apply(
    func=fix_cwa, args=(number_to_country, ))

#>
display(data[feature_name].unique())

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CWA.Row2.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
Here, we just are considering countries that usually someone can apply from, which for Iranians, it's Armenia, Georgia, United Arab Emirates, and Turkey.

Note that, we had CWA cases where the person applied from France, or Iraq. Separately categorizing these might be harder, but we can tell that the person whose CWA is france, already has residency status or travel history that making the case stronger, so other variables dominate affect of this.

*Remark:* My goal for including this feature is that to see if there is bias toward specific countries. I.e. in some Reddit (which is legit) post, people mentioned that UAE cases for non UAE cases usually have higher rate of rejection. 


- [ ] TODO: lets find any dataset/news of tourism visa stats w.r.t. countries.
'''

'''
### P1.PD.CWA.Row2.Status -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CWA.Row2.Status'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper left')
#plt.show()

'''
The interesting observation is that there is huge difference between who choose state of their visit to CWA as VISITOR=3 vs OTHER(Biometric)=6. *I think the reason is that those who choose visitor are way richer on average*, because they need to at least stay there for 2 weeks which costs far more than Biometric visit. Let's see if we are right in the following cells:
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CWA.Row2.Status', 'P1.PD.CWA.Row2.Period'

#nb>
## NBVAL_IGNORE_OUTPUT
#pd.crosstab([data[feature_name[0]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name]], data[feature_name[1]],
#            margins=True, dropna=False).style.background_gradient(cmap='summer_r')

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CWA.Row2.Status', 'P1.PD.CWA.Row2.Period'
display(data.groupby([*feature_name, output_name])[output_name].count())

'''
### Fill CWA NaN and 0 periods with a heuristic
We could go with average, but apparently data is more like discrete where applicants stayed in months rather than weeks/days. Hence, `mode` is a better feature than `mean` as it falls to dominating value.
'''

#>
display(f'{data[feature_name[1]].isna().sum()} `None`s in `{feature_name[1]}`')
display(f'{(data[feature_name[1]] == 0).sum()} `0`s in `{feature_name[1]}`')

#>
# use `.item()` after calling `.mode()` on a dataframe. This is because `.mode()` returns a series, not a scalar!
data.loc[(data[feature_name[1]] == 0.) | (data[feature_name[1]].isna()), feature_name[1]] = data[data[feature_name[1]] != 0][feature_name[1]].mode().item()
display(data[feature_name[1]].unique())

#>
display(f'{data[feature_name[1]].isna().sum()} `None`s in `{feature_name[1]}`')
display(f'{(data[feature_name[1]] == 0).sum()} `0`s in `{feature_name[1]}`')

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#sns.violinplot(x=feature_name[0], y=feature_name[1], hue=output_name, scale='width',
#               data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], split=True, ax=ax)
#ax.set_title('{} and {} vs {}'.format(
#    feature_name[0], feature_name[1], output_name))
#ax.set_yticks(range(0, 120, 10))
#ax.legend(loc='upper left')
#plt.show()

'''
Well, I WAS WRONG. For some reason, where CWA status is `'OTHER'`, people are mostly staying for 90 days, meanwhile for status of Visitor, the duration is mostly around 30 days. The reason it's weird is that usually CWA status of `'OTHER'` stands for `'Biometric'` case and 90 days mode does not make sense!

TODO: what's going on? ask agents!
'''

'''
### P1.PD.CWA.Row2.Other -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CWA.Row2.Other'
display(data.groupby([feature_name, 'P1.PD.CWA.Row2.Status'])[output_name].count())

'''
All the cases that have flag `'P1.PD.CWA.Row2.Other'=True` are in  `'P1.PD.CWA.Row2.Status'=6` which means status of 6 is already representing other case. So, we delete it.
'''

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.MS.SecA.MS -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.MS.SecA.MS'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# convert numbers to names
ms_num_names = {
    '01': 'married',
    '02': 'single',
    '03': 'common-law',
    '04': 'divorced',
    '05': 'separated',
    '06': 'widowed',
    '09': 'annulled',
    '00': 'ukn',
}

data[feature_name] = data[feature_name].apply(lambda x: ms_num_names[x]
                                              if x in ms_num_names else 'ukn')

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
It is clear that *single people (necessarily those who have never married) are way more likely to get rejected*, since single people are way more likely to stay. 
'''

'''
### P2.MS.SecA.PrevMarrIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.PrevMarrIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()

'''
We can see that those who have previous marriage have higher chance, but this could be directly because of the age, since someone with previous marriage is probably more like to have higher age, and higher income (well, women get a lot of money for getting divorced for some unknown reason and those who can divorce usually have higher status as it is rarer to happen in traditional families.)
'''

#>
feature_name = 'P2.MS.SecA.PrevMarrIndicator', 'P1.PD.Sex.Sex'

#nb>
## NBVAL_IGNORE_OUTPUT
#pd.crosstab([data[feature_name[0]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name]], data[feature_name[1]],
#            margins=True, dropna=False).style.background_gradient(cmap='summer_r')

'''
### P2.MS.SecA.TypeOfRelationship -> categorical
'''

#>
# convert numbers to names
ms_num_names = {
    '01': 'married',
    '02': 'single',
    '03': 'common-law',
    '04': 'divorced',
    '05': 'separated',
    '06': 'widowed',
    '09': 'annulled',
    '00': 'ukn',
}

output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.TypeOfRelationship'

# keep it 'ukn' since we have this category in XFA PDFs too
#   although here it means that there was no previous marriage.
data[feature_name] = data[feature_name].apply(lambda x: ms_num_names[x]
                                              if x in ms_num_names else 'ukn')  

display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()

'''
Since we have only type of `'married'`, `'ukn'` represents cases where there was no previous marriage. Hence, feature in previous section, i.e. `'P2.MS.SecA.PrevMarrIndicator'` is representing the exact same thing. On the other hand, common-law marriage is not legal in Iran and as we can see, there is not a single case of it. So even in future, we can just ignore it or replace it with married case. So, we can delete it.
'''

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.MS.SecA.Psprt.CountryofIssue.CountryofIssue -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.Psprt.CountryofIssue.CountryofIssue'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.MS.SecA.Langs.languages.nativeLang.nativeLang -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.Langs.languages.nativeLang.nativeLang'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.MS.SecA.Langs.languages.ableToCommunicate.ableToCommunicate -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.Langs.languages.ableToCommunicate.ableToCommunicate'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()

'''
The data is too small to make a decision, but we see a large acceptance rate when applicant speaks no language, and the reason easily could be that the person is probably old and have a full family, i.e. who can afford to go to Canada but does not speak a second language? Old housewives and retired employees probably! Let's see if we are right for fun:
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Occ.OccRow1.Occ.Occ', 'P2.MS.SecA.Langs.languages.ableToCommunicate.ableToCommunicate'
display(data[data[feature_name[1]] == 'Neither'].groupby([*feature_name, output_name])[output_name].count())

'''
We were right, KEKW. But it has no useful information since it is directly affected by age, family status, etc. Hence, I delete it.
'''

#>
# drop useless
feature_name = 'P2.MS.SecA.Langs.languages.ableToCommunicate.ableToCommunicate'
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.MS.SecA.Langs.LangTest -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.Langs.LangTest'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#plt.show()

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
No idea! Maybe letter we can see some correlation or delete it entirely. For now, we delete it because of few samples. TODO: see what's the best, delete or not?
'''

'''
### P2.natID.q1.natIDIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.natID.q1.natIDIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)
# also this
feature_name = 'P2.natID.natIDdocs.CountryofIssue.CountryofIssue'
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.USCard.q1.usCardIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.USCard.q1.usCardIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.CI.cntct.PhnNums.[Phn.CanadaUS, AltPhn.CanadaUS] -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P2.CI.cntct.PhnNums.Phn.CanadaUS'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

#>
output_name = 'VisaResult'
feature_name = 'P2.CI.cntct.PhnNums.AltPhn.CanadaUS'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#data[data[feature_name] == True]

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
These two entries are the family of the same person with strong application. Although, indication of Canada/US phone depicts strong case, other factors should indicate the same thing too, like previous travel to Canada/US, funds, relationship in Canada, etc. Hence, I remove it for now until further notice!

TODO: see if del or not
'''

'''
### P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# convert numbers to names
vp_num_names = {
    1: 'business',
    2: 'tourism',
    3: 'other',
    4: 'short study',
    5: 'returning student',
    6: 'returning worker',
    7: 'super visa',
    8: 'family visit',
    13: 'visit',
}

output_name = 'VisaResult'
feature_name = 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'
# keep it 'other' since we have this category in XFA PDFs too (as '03')
#   although here it means that there was no previous marriage.
data[feature_name] = data[feature_name].apply(lambda x: vp_num_names[int(x)]
                                              if int(x) in vp_num_names else 'other')

display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#sns.countplot(x=feature_name, hue=output_name, data=data,
#              ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
Clearly, a candidate visiting his family (hence having invitation letter too) has a stronger application than just tourism which is open to all. Also, just `'visit'` is not good enough even though it's not `'tourism'`.
'''

#>
feature_name = 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit', 'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe'

#nb>
## NBVAL_IGNORE_OUTPUT
#z = data[data[feature_name[0]] == 'visit']
#pd.crosstab([z[feature_name[0]], z[(z[output_name] == 'acc') | (z[output_name] == 'rej')][output_name]], z[feature_name[1]],
#            margins=True, dropna=False).style.background_gradient(cmap='summer_r')

'''
Well, yes, of course you get rejected if you are going to spend 10K CAD to see your wife's first cousin. OMEGALUL.
'''

'''
### P3.DOV.PrpsRow1.Other.Other -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.DOV.PrpsRow1.Other.Other'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
display(data[data[feature_name] == True].groupby(
    ['P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit', feature_name])[feature_name].count())

#>
# drop useless
data.drop(feature_name, axis=1, inplace=True)

'''
### P3.DOV.PrpsRow1.Funds.Funds -> Continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.DOV.PrpsRow1.Funds.Funds'

#nb>
#display(data[feature_name].describe())
#
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#sns.histplot(data[feature_name], ax=ax, kde=True)
#ax.set_title('{}'.format(feature_name))
#plt.show()

#nb>
#plt.figure(figsize=(22, 12))
#
#sns.histplot(data[data[output_name] == 'acc'][feature_name],
#             bins=50, color='r', kde=True)
#sns.histplot(data[data[output_name] == 'rej'][feature_name],
#             bins=50, color='g', kde=True)
#plt.title('{} by {}'.format(feature_name, output_name), fontsize=20)
#plt.xlabel(feature_name, fontsize=15)
#plt.ylabel('Density', fontsize=15)
#plt.show()

'''
### P3.DOV.cntcts_Row[i].RelationshipToMe.RelationshipToMe -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe', 'P3.cntcts_Row2.Relationship.RelationshipToMe'

#>
display(data[feature_name[0]].unique())
display(data[feature_name[1]].unique())

'''
We need to categorize these into simpler and representative categories:
1. `['brother', 'sister', 'step*', ...]: 'f1'` 
2. `['*in law*', 'nephew']: 'f2'`
3.  `'*friend*': 'friend'`


Issues:
1. Should we put `'hotel'` first or `'cousin'` in `'The Husband of My Wife\\\\s first cousin/hotel'`?  TODO: ask agents which one has higher priority
'''

#>
rel_cat = {  # order matters, put weaker on top, i.e. put 'law' above 'brother', so 'brother in law' get handled by 'law' rule than 'bother' rule
    'law': 'f2',
    'nephew': 'f2',
    'niece': 'f2',
    'aunt': 'f2',
    'uncle': 'f2',
    'cousin': 'f2',
    'relative': 'f2',
    'grand': 'f2',
    'parent': 'f1',
    'mother': 'f1',
    'father': 'f1',
    'child': 'f1',
    'daughter': 'f1',
    'brother': 'f1',
    'sister': 'f1',
    'wife': 'f1',
    'husband': 'f1',
    'step': 'f1',
    'son': 'f1',
    'partner': 'f1',
    'fiance': 'f1',
    'fiancee': 'f1',
    'other': 'ukn',
    'friend': 'friend',
    'league': 'work',
    'symposium': 'work',
    'hote': 'hotel',
    'hotel': 'hotel',
}


def fix_rel(string: str, dic: dict):
    string = string.lower()
    for k, v in dic.items():
        if k in string:
            string = string.replace(string, v)
            return string
    return 'ukn'


data[feature_name[0]] = data[feature_name[0]].apply(fix_rel, args=(rel_cat, ))
data[feature_name[1]] = data[feature_name[1]].apply(fix_rel, args=(rel_cat, ))

#>
display(data[feature_name[0]].unique())
display(data[feature_name[1]].unique())

#nb>
#f, ax = plt.subplots(2, 2, figsize=(26, 12))
#data[[feature_name[0], output_name]].groupby(
#    [feature_name[0]]).count().plot.bar(ax=ax[0, 0], sort_columns=True)
#add_percentage_axes(ax[0, 0], len(data))
#ax[0, 0].set_title('{} vs {}'.format(output_name, feature_name[0]))
#sns.countplot(x=feature_name[0], hue=output_name, data=data,
#              ax=ax[0, 1], hue_order=output_hue_order)
#add_percentage_axes(ax[0, 1], len(data))
#ax[0, 1].set_title('{}: {}'.format(feature_name[0], output_name))
#
#data[[feature_name[1], output_name]].groupby(
#    [feature_name[1]]).count().plot.bar(ax=ax[1, 0], sort_columns=True)
#add_percentage_axes(ax[1, 0], len(data))
#ax[1, 0].set_title('{} vs {}'.format(output_name, feature_name[1]))
#sns.countplot(x=feature_name[1], hue=output_name, data=data,
#              ax=ax[1, 1], hue_order=output_hue_order)
#add_percentage_axes(ax[1, 1], len(data))
#ax[1, 1].set_title('{}: {}'.format(feature_name[1], output_name))
#ax[1, 1].legend(loc='upper right')
#plt.show()

'''
### P3.Edu.EduIndicator -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Edu.EduIndicator'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#ax[1] = sns.countplot(x=feature_name, hue=output_name, data=data,
#                      ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper left')
#plt.show()

'''
It seems that around %40 of people without education got accepted but %37.5 of people with education got accepted. These percentages are close and other factors such as age might have greater effect.

Also, I think the education field has way more effect than just having higher education. Of course we prefer a engineer/medic refugee than some pleb that going to consume more than provide.

TODO: somehow categorize education level/field into an ordered list, then show the acceptance rate given the education level/field.
'''

'''
### P3.Edu.Edu_Row1.Country.Country -> categorical -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Edu.Edu_Row1.Country.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

'''
Dealing with countries in categorical form is not logical. We would like to rank them or in an easier way, have a continuous score for each country. This way, we can convert a possibly large categorical feature into continuous field.

To do so, approach discussed in issue #10 (PR #11) has been taken, which in summary, takes WorldBank data, looks for education fields, and take average of countries score over past years. Here, we utilize those methods.
'''

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME
data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

'''
For some reason, I filled all `None`'s with `'IRAN'` which is a wrong action. For now, I fix it by setting it to `None` by overlapping `'P3.Edu.Edu_Row1.Country.Country' ` and `'P3.Edu.EduIndicator'`
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Edu.Edu_Row1.Country.Country' 

display(data[data['P3.Edu.EduIndicator'] == False].__len__() -
      data[data[feature_name].isna()].__len__())  # must be = 0

data.loc[data['P3.Edu.EduIndicator'] == False, feature_name] = None

display(data[data['P3.Edu.EduIndicator'] == False].__len__() -
      data[data[feature_name].isna()].__len__())  # must be = 0

'''
Time to convert country names to continuous values of scores of them.
'''

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_OVERALL, repo=REPO, rev=HELPER_VERSION_OVERALL))
edu_country_score_preprocessor = preprocessor.EducationCountryScoreDataframePreprocessor(
    dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=edu_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[data['P3.Edu.EduIndicator'] == False][feature_name].unique())

#>
# convert to years but continuous
feature_name = 'P3.Edu.Edu_Row1.Period'
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

#nb>
#data.sample(1, random_state=SEED)

'''
### P3.Edu.Edu_Row1.FieldOfStudy -> categorical/ordered/continuous
'''

'''
TODO: can we do continuous here?

'''

'''
There are different ways again to deal with this, ranking, continuous, etc.

Here are the methods:
1. **Ranking** only based on the higher education level, i.e. `[apprentices=1, worker diploma=2, bachelor=3, master=5, phd=8`]. (we entirely ignore the field actually, just level of education). ([ref](https://www.canada.ca/en/immigration-refugees-citizenship/services/application/application-forms-guides/guide-5256-applying-visitor-visa-temporary-resident-visa.html)) **<-- CURRENT IMPL.**
2. *manually* ranking field of study and multiply it by method (1); i.e. `new = result1 * result2`


***Remark: This section assumes applicants' highest post secondary education**. I.e. if you have a bachelor and masters, you have to put masters and its period which is 2 years rather than 6 years (4 years of bachelor and 2 years of masters)*
'''

#>
from enum import Enum
class field_of_study_rank(Enum):
    apprentice = 1  # 1*1 = 1
    diploma = 2  # 2*2 = 4
    bachelor = 3  # 3*4 = 12
    master = 8  # 8*2 = 16
    phd = 10  # 10*3 = 30

#>
output_name = 'VisaResult'
feature_name = 'P3.Edu.Edu_Row1.FieldOfStudy', 'P3.Edu.Edu_Row1.Period', 'P3.Occ.OccRow1.Occ.Occ'

'''
#### `'P3.Edu.Edu_Row1.Period' > 8`
'''

#>
cond = (data[feature_name[1]] >= 8) & (data[feature_name[1]] <= 99)
display(data.loc[cond, feature_name[0]])
data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(lambda x: 'phd')
display(data.loc[cond, feature_name[0]])

'''
#### `7 <= 'P3.Edu.Edu_Row1.Period' < 8`
'''

#>
cond = (data[feature_name[1]] >= 7) & (data[feature_name[1]] < 8)
display(data.loc[cond, feature_name[0]])
data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(lambda x: 'phd' if 'med' in x.lower() else 'master')
display(data.loc[cond, feature_name[0]])

'''
#### `6 <= 'P3.Edu.Edu_Row1.Period' < 7`
'''

#>
cond = (data[feature_name[1]] >= 6) & (data[feature_name[1]] < 7)
display(data.loc[cond, feature_name[0:2]])
data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(lambda x: 'master')
display(data.loc[cond, feature_name[0]])

'''
#### `3.5 <= 'P3.Edu.Edu_Row1.Period' < 6`
'''

#>
cond = (data[feature_name[1]] >= 3.5) & (data[feature_name[1]] < 6)
display(data.loc[cond, feature_name[0:2]])
data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(lambda x: 'bachelor')
display(data.loc[cond, feature_name[0]])

'''
#### `3 <= 'P3.Edu.Edu_Row1.Period' < 3.5`
'''

#>
cond = (data[feature_name[1]] >= 3) & (data[feature_name[1]] < 3.5)
display(data.loc[cond, feature_name[0:2]])
data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(lambda x: 'master')
display(data.loc[cond, feature_name[0]])

'''
#### `2 <= 'P3.Edu.Edu_Row1.Period' < 3`
'''

#>
cond = (data[feature_name[1]] >= 2) & (data[feature_name[1]] < 3)
display(data.loc[cond, feature_name[0:2]])

def field_of_study_converter(x: str) -> str:
    _keys = ['logy', 'eng', 'manag', 'mb', 'med', 'sci', 'adm', 'law', 'physic', ]
    val = 'diploma'
    for k in _keys:
        if k in x.lower():
            val = 'master'
    return val

data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(field_of_study_converter)
display(data.loc[cond, feature_name[0]])

'''
#### `0 < 'P3.Edu.Edu_Row1.Period' < 2`
'''

#>
cond = (data[feature_name[1]] > 0) & (data[feature_name[1]] < 2)
display(data.loc[cond, feature_name[0:2]])
def field_of_study_converter(x: str) -> str:
    _keys = ['mba', 'admin', 'med', 'scien']
    val = 'apprentice'
    for k in _keys:
        if k in x.lower():
            val = 'master'
    return val

data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(field_of_study_converter)
display(data.loc[cond, feature_name[0]])

'''
#### Missed and Uneducated
'''

'''
1. **uneducated**: Those who do not have field of study and have zero education period
2. **missed**: Those who have > 0 education period but no field of study
'''

#>
display(f'{len(data[(data[feature_name[0]].isna())])} cases with missing field of study')
display(f'{data[(data[feature_name[0]].isna()) & (data[feature_name[1]] == 0.)][[*feature_name]].__len__()} case are uneducated.')

#>
display(data[(data[feature_name[0]].isna()) & (data[feature_name[1]] == 0.)][[*feature_name]].sample(10, random_state=SEED))

'''
Fill all **uneducated**'s field of study with with `'unedu'` just for readability purposes.
'''

#>
cond = (data[feature_name[0]].isna()) & (data[feature_name[1]] == 0.)
data.loc[cond, feature_name[0]] = data.loc[cond, feature_name[0]].apply(lambda x: 'unedu')

#>
display(data[(data[feature_name[1]] == 0.)][[*feature_name]].sample(10, random_state=SEED))

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name[0], output_name]].groupby(
#    [feature_name[0]]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name[0]))
#ax[1] = sns.countplot(x=feature_name[0], hue=output_name, data=data,
#                      ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name[0], output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
Insights:
1. There is a slight positive rate towards `master` in comparison to `bachelor`.
2. `phd` (`phd` and `md`) dominates but having acceptance above 0.5.
3. `unedu` is almost representative of all cases, since almost majority of `unedu` are `housewife` cases and their husband's application is strongly affecting

TODO: for creating dataset, I have treated each member of the family as an separate application and I am supposed to only copy-paste `other.csv` part between members of the same family. But for cases when the *female's job is housewife*, features `P3.Edu` and `P3.Occ` need to be updated respectively. I.e. we should not have `housewife` case at all unless main applicant was `housewife` which sounds impossible. <span style="color: cyan"> #46 </span> 
'''

'''
### P3.Occ.OccRow[1,2,3].Occ.Occ -> categorical -> continuous
'''

'''
This is hard to deal with too, can we assign continuous values? can we rank and categorize into 5, 10 categories?

Categories that are considered and filled by keyword matching:
1. manager: `['manager', 'chair', 'business', 'direct', 'owner', 'share', 'board', 'head', 'ceo']`
2. specialist: `['eng', 'doc', 'med', 'arch', 'expert']`
3. employee: `['sale', 'employee', 'teacher', 'retail']` and anything else except `'OTHER'` (`'OTHER'` is used for `None` jobs)
4. retired: `['retire']`
5. student: `['student', 'intern']`
6. house wife: `['wife']`
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Occ.OccRow1.Occ.Occ', 'P3.Occ.OccRow2.Occ.Occ', 'P3.Occ.OccRow3.Occ.Occ'

#>
unique_occ = np.array([]).astype('object')
for f in feature_name:
    unique_occ = np.concatenate([unique_occ, data[f].unique()])
display(unique_occ.__len__())
display(unique_occ[:10])

#>
# order is important, e.g. `'student of computer engineering'` should be categorized as `'student'` not `'specialist'` (because of `'eng'`)
occ_cat_dict = {
    'manager': ['manager', 'chair', 'business', 'direct', 'owner', 'share', 'board', 'head', 'ceo'],
    'student': ['student', 'intern'],
    'retired': ['retire'],
    'specialist': ['eng', 'doc', 'med', 'arch', 'expert'],
    'employee': ['sale', 'employee', 'teacher', 'retail'],
    'housewife': ['wife'],
}


def categorize_occ(x: str, d: dict, default='employee') -> str:
    """if `x` found in any of `d.value`, return the corresponding `d.key`.

    Args:
        x (str): input string to search for
        d (dict): the dictionary to look for `x` in its values and return the key
        default (str, optional): if `x` no found in `d` at all. Defaults to 'employee'.
            except cases mentioned in `d` or `'OTHER'` 

    Returns:
        str: string containing a key in `d`
    """
    x = x.lower()

    # find occurrences
    for k in d.keys():
        d_vals = d[k]
        found = len([v for v in d_vals if v in x]) > 0
        if found:
            return k
    return default if x != 'other' else 'OTHER'


display(data.loc[:, [*feature_name]])  # old
data.loc[:, [*feature_name]] = data.loc[:, [*feature_name]].applymap(categorize_occ,
                                                                     d=occ_cat_dict,
                                                                     default='employee', )
display(data.loc[:, [*feature_name]])  # new

#>
feature_name = 'P3.Occ.OccRow1.Occ.Occ', 'P3.Occ.OccRow2.Occ.Occ', 'P3.Occ.OccRow3.Occ.Occ'
_feature_name = 'P3.Occ.OccRow1.Period', 'P3.Occ.OccRow2.Period', 'P3.Occ.OccRow3.Period'

#nb>
#f, ax = plt.subplots(2, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[0], y=_feature_name[0], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0, 0])
#ax[0, 0].set_title('{} and {} vs {}'.format(feature_name[0], _feature_name[0], output_name))
## ax[0, 0].set_ylim(0, 15000)
#sns.violinplot(x=feature_name[1], y=_feature_name[1], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0, 1])
#ax[0, 1].set_title('{} and {} vs {}'.format(feature_name[1], _feature_name[1], output_name))
## ax[0, 1].set_ylim(0, 15000)
#sns.violinplot(x=feature_name[2], y=_feature_name[2], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[1, 0])
#ax[1, 0].set_title('{} and {} vs {}'.format(feature_name[2], _feature_name[2], output_name))
## ax[1, 0].set_ylim(0, 15000)
## ax[0, 0].set_yticks(range(0, 200, 10))
#plt.show()
#

'''
Insights:
1. in almost all cases we can see that higher the period, higher the chance. This is because of age too, higher the work period means person has higher age. So, this is a correlation that was expected. Most obvious case of this scenario is the `housewife` case where practically person is unemployed (worst case) and the longer the person is unemployed the higher chance he/she has which is direct result of age factor.
2. In case of `employee`s and `specialist`s, we can see that the effect of age becomes way less.
3. Those whose *previous* job were `manager` or `specialist` (which may sound that they are not longer in that job) have way higher rejection. I.e. if you had more responsibility before your current job, you are less likely to be satisfied with your *current* status.
'''

'''
### P3.Occ.OccRow1.Country.Country -> categorical -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Occ.OccRow1.Country.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

'''
Dealing with countries in categorical form is not logical. We would like to rank them or in an easier way, have a continuous score for each country. This way, we can convert a possibly large categorical feature into continuous field.

To do so, approach discussed in issue #10 (PR #11) has been taken, which in summary, takes WorldBank data, looks for economical fields (GDP), and take average of countries score over past years. Here, we utilize those methods.
'''

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME
data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

'''
Time to convert country names to continuous values of scores of them.
'''

#>
display(data[feature_name].unique())

'''
options: 
- 'GCI 4.0: Global Competitiveness Index 4.0, Rank'
- [NEW] GDP per capita: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD 
'''

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP))
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=eco_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[feature_name].unique())

'''
#### P3.Occ.OccRow2.Country.Country -> categorical -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Occ.OccRow2.Country.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME
data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP))
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=eco_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[feature_name].unique())

'''
#### P3.Occ.OccRow3.Country.Country -> categorical -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.Occ.OccRow3.Country.Country'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME
data[feature_name] = data[feature_name].apply(func=functional.extended_dict_get, args=(
    functional.config_csv_to_dict(config_path), 'Unknown', str.isnumeric, ))

#>
tmp_df = pd.read_pickle(dvc.api.get_url(path=HELPER_PATH_GDP, repo=REPO, rev=HELPER_VERSION_GDP))
eco_country_score_preprocessor = preprocessor.WorldBankXMLProcessor(dataframe=tmp_df)
data[feature_name] = data[feature_name].apply(
    func=eco_country_score_preprocessor.convert_country_name_to_numeric)

#>
display(data[feature_name].unique())

'''
#### P3.Occ.OccRowX.Period -> continuous
'''

#>
r = re.compile('P3.Occ.OccRow.*.Period')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

'''
### P3.BGI.Details.MedicalDetails -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.BGI.Details.MedicalDetails'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
feature_name = 'P3.BGI.Details.MedicalDetails'
data.drop(feature_name, axis=1, inplace=True)

'''
### P3.BGI.otherThanMedic -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.BGI.otherThanMedic'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
feature_name = 'P3.BGI.otherThanMedic'
data.drop(feature_name, axis=1, inplace=True)

'''
### P3.noAuthStay -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.noAuthStay'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#ax[1] = sns.countplot(x=feature_name, hue=output_name, data=data,
#                      ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
Obviously, staying in a country beyond authenticated duration, dominates solely and even though it sounds obvious, it shouldn't be ignored.
'''

'''
### P3.refuseDeport -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.refuseDeport'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#ax[1] = sns.countplot(x=feature_name, hue=output_name, data=data,
#                      ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
Of course, those who have previous rejected cases are more likely to get rejected again.
'''

'''
### P3.BGI2.PrevApply -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.BGI2.PrevApply'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#ax[1] = sns.countplot(x=feature_name, hue=output_name, data=data,
#                      ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
Well, it sounds obvious that if someone got the visa before, it is more likely to get the visa once again and vice versa. So, let's compare with `'P3.refuseDeport'` to see the relation between previous applies that succeeded or failed.
'''

#>
feature_name = 'P3.BGI2.PrevApply', 'P3.refuseDeport'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) *100
#ct

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 15))
#data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name].value_counts().plot.pie(
#    autopct='%1.2f%%', ax=ax[0], shadow=True, textprops={'color': 'black'})
#ax[0].set_title('VisaResult')
#ax[0].set_ylabel('')
#
#ct.plot(kind='bar', stacked=True, rot=0, ax=ax[1])
#ax[1].set_title('{} and {}: {}'.format(
#    feature_name[0], feature_name[1], output_name))
#ax[1].legend(loc='upper left')
#for c in ax[1].containers:
#    ax[1].bar_label(c, label_type='center', fmt='%1.1f%%')
#plt.show()

'''
First of all, we can see that %38.7 of applicants got visa. Now, let's see how many of them got it because of their previous apply.

Insights:
1. As we can see from the 3rd column, around %14.5 of applicant got visa because they had applied previously and got accepted. 
2. Interestingly, around %6.5(=2.8+3.7) of applicant got visa, even though they have been previously rejected.
3. Other %17.8 got visa without any prior record.
4. Having visa rejection (2nd column) reduces the chance of acceptance by at least 2 times!
'''

'''
### P3.PWrapper.criminalRec -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.PWrapper.criminalRec'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
feature_name = 'P3.PWrapper.criminalRec'
data.drop(feature_name, axis=1, inplace=True)

'''
### P3.PWrapper.Military.Choice -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.PWrapper.Military.Choice'
display(data.groupby([feature_name, output_name])[output_name].count())

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#data[[feature_name, output_name]].groupby(
#    [feature_name]).count().plot.bar(ax=ax[0])
#add_percentage_axes(ax[0], len(data))
#ax[0].set_title('{} vs {}'.format(output_name, feature_name))
#ax[1] = sns.countplot(x=feature_name, hue=output_name, data=data,
#                      ax=ax[1], hue_order=output_hue_order)
#add_percentage_axes(ax[1], len(data))
#ax[1].set_title('{}: {}'.format(feature_name, output_name))
#ax[1].legend(loc='upper right')
#plt.show()

'''
It's nothing to consider. If all was rejected, we could say it is a warning, but apparently has no effect. Especially, many have done compulsory service. Hence, for now we drop it.
'''

#>
# drop useless
feature_name = 'P3.PWrapper.Military.Choice'
data.drop(feature_name, axis=1, inplace=True)

'''
### P3.PWrapper.politicViol -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.PWrapper.politicViol'
display(data.groupby([feature_name, output_name])[output_name].count())

'''
So ... Yea... everybody lied that they have not seen any mistreatment! KEKSociety.
'''

#>
# drop useless
feature_name = 'P3.PWrapper.politicViol'
data.drop(feature_name, axis=1, inplace=True)

'''
### P3.PWrapper.witnessIllTreat -> categorical
'''

#>
output_name = 'VisaResult'
feature_name = 'P3.PWrapper.witnessIllTreat'
display(data.groupby([feature_name, output_name])[output_name].count())

#>
# drop useless
feature_name = 'P3.PWrapper.witnessIllTreat'
data.drop(feature_name, axis=1, inplace=True)

'''
### P1.PD.CurrCOR.Row2.Period -> continuous
'''

'''
This part is a little different, since most of the people are single-national citizens (in our domain at least), then their `'CurrCOR'` and `'DOBYear.Period'` are the same value.
So we prefer to compare these to together as a joint variable. 
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.CurrCOR.Row2.Period', 'P1.PD.DOBYear.Period'

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.kdeplot(x=feature_name[0], hue=output_name, common_norm=False,
#            data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], fill=True, ax=ax[0])
#ax[0].set_title('{} vs {}'.format(feature_name[0], output_name))
#sns.kdeplot(x=feature_name[1], hue=output_name, common_norm=False,
#            data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], fill=True, ax=ax[1])
#ax[1].set_title('{} vs {}'.format(feature_name[1], output_name))
#plt.show()

'''
As we can see, both data has the same distribution except a small proportion (around 0) which can be considered as anomaly. Hence, I think dropping this `'CurrCOR'` is a rational choice. But to make sure, we plot these these two variables against each other.
'''

#>
# drop useless
feature_name = 'P1.PD.CurrCOR.Row2.Period'
data.drop(feature_name, axis=1, inplace=True)

'''
### 'P1.PD.DOBYear.Period' -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.DOBYear.Period'

#>
# convert to years but continuous
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

#nb>
#display(data[data[feature_name]<=25].groupby([feature_name, output_name])[output_name].count())

'''
Let's get rid of that kid first. But I decided to keep others (e.g. 17, 18, ... years old kids) and the reason is that they have been proportionately rejected. So, if they wanted to go alone, or with family, there is something that caused them to get rejected. So, let's keep em.
'''

#>
data.drop(data[data[feature_name]<=1].index, inplace=True)
display(data[data[feature_name]<=1][feature_name])

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#sns.kdeplot(x=feature_name, hue=output_name, common_norm=False,
#            data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], fill=True, ax=ax)
#ax.set_title('{} vs {}'.format(feature_name, output_name))
#ax.set_xticks(range(0, 100, 5))
#plt.show()

'''
### 'P1.PD.DOBYear.Period' and a second feature
'''

'''
#### 'P1.PD.DOBYear.Period' and 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.DOBYear.Period', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit', 'P1.PD.Sex.Sex'

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[1], y=feature_name[0], scale='area', 
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0])
#ax[0].set_title('{} and {} vs {}'.format(feature_name[1], feature_name[0], output_name))
#ax[0].set_yticks(range(0, 110, 10))
#sns.violinplot(x=feature_name[2], y=feature_name[0], scale='width', 
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[1])
#ax[1].set_title('{} and {} vs {}'.format(feature_name[2], feature_name[0], output_name))
#ax[1].set_yticks(range(0, 110, 10))
#plt.show()

'''
Insights:
1. The lower the age, the lower the likelihood
2. 'family visit' dominates the age and increases the likelihood considerably
3. There is not much difference in 'tourism' and 'visit' and 'age' is the the important factor again
4. The peak of rejection for 'Male' is around 40 while for 'Female' is around 30 and the reason is that usually in families, men have higher age than women.

'''

'''
#### 'P1.PD.DOBYear.Period' and 'P3.DOV.PrpsRow1.Funds.Funds' -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.DOBYear.Period', 'P3.DOV.PrpsRow1.Funds.Funds'

#nb>
#sns.jointplot(
#    data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], x=feature_name[0], y=feature_name[1], hue=output_name, height=12)
#plt.ylim(0, 12e3)
#plt.show()

'''
Since lower age decreases the likelihood of acceptance considerably, having larger funds is compensating.
'''

'''
#### 'P1.PD.DOBYear.Period' and 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.DOBYear.Period', 'P1.MS.SecA.MS'

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[1], y=feature_name[0], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0])
#ax[0].set_title('{} and {} vs {}'.format(feature_name[1], feature_name[0], output_name))
#ax[0].set_yticks(range(0, 110, 5))
#sns.kdeplot(x=feature_name[0], hue=feature_name[1], common_norm=False,
#            data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], fill=True, ax=ax[1])
#ax[1].set_title('{} vs {}'.format(feature_name[0], feature_name[1]))
## ax[1, 1].set_xticks(range(0, 100, 5))
#plt.show()
#

'''
It seems that 'marital status' has been encoded in 'age' or wise versa.

For instance:
1. Single people's acceptance mean is way lower age than other groups. This is counter intuitive, because most of these young and 'single' people are with their family probably, so their parents are the actual applicants. This can be confirmed by looking at the trail of 'single' where it expands to lower ages.
2. 'divorced' and 'married' people's age seems irrelevant. Although, we see that peak of rejection and acceptance for 'divorced' people are higher and this is obvious since usually, 'divorced' people have higher average age.
3. 'widowed' case has similar status as the latter statement in 2.
'''

'''
### 'P1.MS.SecA.DateOfMarr.Period' -> continuous
'''

#>
# convert to years but continuous
feature_name = 'P1.MS.SecA.DateOfMarr.Period'
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

#>
output_name = 'VisaResult'
feature_name = 'P1.MS.SecA.DateOfMarr.Period', 'P1.PD.Sex.Sex'

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[1], y=feature_name[0], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0])
#ax[0].set_title('{} and {} vs {}'.format(feature_name[1], feature_name[0], output_name))
#ax[0].set_yticks(range(0, 80, 5))
#sns.kdeplot(x=feature_name[0], hue=feature_name[1], common_norm=False,
#            data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], fill=True, ax=ax[1])
#ax[1].set_title('{} vs {}'.format(feature_name[0], feature_name[1]))
#plt.show()

'''
Seems like there is no insight in relation of gender to marriage period.
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.MS.SecA.DateOfMarr.Period', 'P1.PD.DOBYear.Period'

#nb>
#g= sns.jointplot(
#    data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], x=feature_name[0], y=feature_name[1], hue=output_name, height=12)
#g.ax_joint.set_xticks(range(0, 81, 5))
#g.ax_joint.set_yticks(range(15, 90, 5))
#plt.show()

#>
output_name = 'VisaResult'
feature_name = 'P1.MS.SecA.DateOfMarr.Period', 'P1.PD.Sex.Sex', 'P1.PD.DOBYear.Period', 

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[1], y=feature_name[0],
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0])
#ax[0].set_title('{} and {} vs {}'.format(feature_name[1], feature_name[0], output_name))
#ax[0].set_yticks(range(0, 80, 5))
#sns.violinplot(x=feature_name[1], y=feature_name[2],
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[1])
#ax[1].set_title('{} vs {}'.format(feature_name[2], feature_name[1]))
#ax[1].set_yticks(range(0, 100, 5))
#plt.show()

'''
Even though it seems that a lot of people with short to medium marriages (<=10) when they are in their early 30s or late 20s have been rejected (previous scatter graph), here, we can see that having marriage even for a short period increases the likelihood considerably (look at the bumps on the figure on the left around Yaxis=0 and smooth very low density on the figure on the right.)

TODO: see issue #15. Seems that such a thing would improve.
'''

'''
### 'P2.MS.SecA.PrevSpouseDOB.DOBYear.Period' and 'P2.MS.SecA.Period' -> continuous
'''

#>
# convert to years but continuous
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.PrevSpouseDOB.DOBYear.Period', 'P2.MS.SecA.Period'
data[feature_name[0]] = data[feature_name[0]].apply(lambda x: x/365.)
data[feature_name[1]] = data[feature_name[1]].apply(lambda x: x/365.)

#>
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.PrevSpouseDOB.DOBYear.Period', 'P1.MS.SecA.MS', 'P2.MS.SecA.Period'

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[1], y=feature_name[0],
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0])
#ax[0].set_title('{} and {} vs {}'.format(feature_name[1], feature_name[0], output_name))
#ax[0].set_yticks(range(0, 100, 5))
#sns.violinplot(x=feature_name[1], y=feature_name[2],
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[1])
#ax[1].set_title('{} vs {}'.format(feature_name[2], feature_name[1]))
#ax[1].set_yticks(range(0, 100, 5))
#plt.show()

'''
It is hard to extract any insights from it since all range of age exist for all categories (except single/married of course), so I leave it there until further investigation.

TODO: maybe delete? no idea how useful it is 
1. Having previous marriage's period seems to contain all information we need about previous marriage, hence, the age of previous spouse does not provide any useful information.
'''

#>
# drop useless
feature_name = 'P2.MS.SecA.PrevSpouseDOB.DOBYear.Period'
data.drop(feature_name, axis=1, inplace=True)

'''
### P2.MS.SecA.Psprt.ExpiryDate.Remaining -> continuous
'''

#>
# convert to years but continuous
output_name = 'VisaResult'
feature_name = 'P2.MS.SecA.Psprt.ExpiryDate.Remaining'
data[feature_name] = data[feature_name].apply(lambda x: x/30.)

#>
feature_name = 'P2.MS.SecA.Psprt.ExpiryDate.Remaining'
display(data[data[feature_name] < 5.].__len__())

'''
For some reason, we have to samples that have negative expiration date. For conversion, we can use `.abs` to mirror them or use statistical methods to fill, e.g. `mode`.
'''

#>
data.loc[data[feature_name] < 5., feature_name] = data[data[feature_name] < 5.][feature_name].apply(lambda x: np.abs(x))
display(data[data[feature_name] < 5.].__len__())

#nb>
#feature_name = 'P2.MS.SecA.Psprt.ExpiryDate.Remaining'
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#sns.kdeplot(x=feature_name, hue=output_name, common_norm=False,
#            data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], fill=True, ax=ax)
#ax.set_xticks(range(-50, 150, 5))
#ax.set_title('{} (m) vs {}'.format(feature_name, output_name))
#plt.show()

'''
There are three points of interest here, one around 5-10 and the other around 60 on x axis.
1. 5-10: It seems that not having long enough expiration date on passport suggests that the applicant is more likely to get rejected.
2. 60: 60 month or 5 years are the expiration period for fresh Iranian passports and it might mean that they are children of a family and their parent's features are carrying them.
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.PD.DOBYear.Period', 'P2.MS.SecA.Psprt.ExpiryDate.Remaining'

#nb>
#g = sns.jointplot(
#    data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')], x=feature_name[0], y=feature_name[1], hue=output_name, height=12)
## g.ax_joint.set_xticks(range(0, 81, 5))
#g.ax_joint.set_yticks(range(0, 125, 5))
#plt.show()

'''
### P3.DOV.PrpsRow1.HLS.Period -> continuous
'''

'''
#### P3.DOV.PrpsRow1.HLS.Period and 'P3.DOV.PrpsRow1.Funds.Funds'
'''

#>
# convert to years but continuous
output_name = 'VisaResult'
feature_name = 'P3.DOV.PrpsRow1.HLS.Period', 'P3.DOV.PrpsRow1.Funds.Funds'

#nb>
#data[data[feature_name[1]] > 10000]

'''
we have two cases where have 20K and 30K funds, hence, for visualization purposes, I have excluded them.
'''

#nb>
#g = sns.jointplot(
#    data=data[((data[output_name] == 'acc') | (data[output_name] == 'rej')) & (data[feature_name[1]] < 10000)],
#    x=feature_name[0], y=feature_name[1], hue=output_name, height=12)
#g.ax_joint.set_xticks(range(0, 200, 10))
#plt.show()

'''
It seems many people who have **chosen below 30 days stay have been rejected for any amount of funds.**. As we can see on the top distribution, peak of rejection is on 10 days!

But, this could be because of other reasons, for instance, those who are coming for tourism, would mostly stay for 2 weeks, and most of the rejections are for tourism. So, we check purpose of visit and the duration.
'''

'''
#### P3.DOV.PrpsRow1.HLS.Period and 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'
'''

#>
# convert to years but continuous
output_name = 'VisaResult'
feature_name = 'P3.DOV.PrpsRow1.HLS.Period', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit', 'P3.DOV.PrpsRow1.Funds.Funds'

#nb>
#f, ax = plt.subplots(1, 2, figsize=(22, 12))
#sns.violinplot(x=feature_name[1], y=feature_name[0], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[0])
#ax[0].set_title('{} and {} vs {} (d)'.format(feature_name[1], feature_name[0], output_name))
#ax[0].set_yticks(range(0, 200, 10))
#sns.violinplot(x=feature_name[1], y=feature_name[2], scale='area',
#               hue=output_name, data=data[(data[output_name] == 'acc') | (data[output_name] == 'rej')],
#               split=True, ax=ax[1])
#ax[1].set_title('{} vs {}'.format(feature_name[2], feature_name[1]))
## ax[1, 1].set_xticks(range(0, 100, 5))
#plt.show()
#

'''
My intuition was not wrong! We can see that even though peak of funds for all visits (except family visit a bit) are around 10K, meanwhile, those who chose shorter visits, i.e. <30 days, have been rejected more. 
'''

'''
### p1.Subform1.Visitor, p1.Subform1.Worker, p1.Subform1.Student, and p1.Subform1.Other
'''

#>
output_name = 'VisaResult'
feature_name = 'p1.Subform1.Visitor', 'p1.Subform1.Worker', 'p1.Subform1.Student', 'p1.Subform1.Other'
display(data.groupby([*feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop([*feature_name], axis=1, inplace=True)

'''
Ok, everyone is a `visitor`, they just forgot to mention it! So, delete it.
'''

'''
### p1.SecA.Sps.SpsCOB, p1.SecA.Fa.FaCOB, p1.SecA.Mo.MoCOB, p1.SecB.Chd.X.ChdCOB, p1.SecC.Chd.X.ChdCOB -> categorical
'''

'''
I don't think having a father born in different country make a difference from having a foreigner mother. Same for brother or sister. So, let's aggregate all of these into one single column, calling it `hasForeignF1` which means if the applicant has any tier 1 family member, with different nationality than applicant's.

***BUT***, there should be difference in following cases:
1. Having a foreign **sibling**: Candidate more likely to not live in their own country of birth


It's better to create a separate variable for cases mentioned above.
'''

#>
feature_name = [c for c in data.columns.values if 'ChdCOB' in c]
feature_name.extend([c for c in data.columns.values if 'MoCOB' in c or 'FaCOB' in c or 'SpsCOB' in c])
display(feature_name)

#>
# fillna with IRAN
display(data[feature_name].isna().sum())
data[feature_name] = data[feature_name].fillna(value='IRAN')

#>
config_path = CANADA_COUNTRY_CODE_TO_NAME
typos = ['iram', 'iaran', 'mahallat-iran', 'astara-iran', 'tehran-iran']

data[feature_name] = data[feature_name].applymap(func=functional.extended_dict_get,
                                                 dic=functional.config_csv_to_dict(config_path),
                                                 if_nan='iran', condition=str.isnumeric)
data[feature_name] = data[feature_name].applymap(func=str.lower)  # conditions for aggs are in lowercase                                             
data[feature_name] = data[feature_name].applymap(func=functional.fix_typo, typos=typos, fix='iran')

'''
#### p1.SecC.Chd.X.ChdCOB -> categorical
'''

'''
Convert the list of siblings' country of birth to a single variable depicting how many foreigner sibling the candidate has.
'''

#>
r = re.compile('p1.SecC.Chd.*.ChdCOB')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
output_name = 'VisaResult'
display(data.groupby([*feature_name, output_name])[output_name].count())

#>
# replace rows of previous country of residency to count of them
agg_column_name = 'p1.SecC.Chd.X.ChdCOB.ForeignerCount'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_foreigner_family,
                                   agg_column_name=agg_column_name, columns=feature_name)
# delete redundant columns tnx to newly created 'p1.SecC.Chd.X.ChdCOB.ForeignerCount'
data.drop(feature_name, axis=1, inplace=True)

#nb>
#data[data[agg_column_name] > 0]

#>
output_name = 'VisaResult'
feature_name = 'p1.SecC.Chd.X.ChdCOB.ForeignerCount', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(2, 1, figsize=(23, 16))
#data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name].value_counts().plot.pie(
#    autopct='%1.2f%%', ax=ax[0], shadow=True, textprops={'color':'black'})
#ax[0].set_title(output_name)
#ax[0].set_ylabel('')
#
#ct.plot(kind='bar', stacked=True, rot=0, ax=ax[1])
#ax[1].set_title('{} and {}: {}'.format(
#    feature_name[0], feature_name[1], output_name))
#ax[1].legend(loc='upper left')
#for c in ax[1].containers:
#    ax[1].bar_label(c, label_type='center', fmt='%1.1f%%', color='black')
#plt.show()

'''
#### p1.SecB.Chd.X.ChdCOB, p1.SecA.Mo.MoCOB, p1.SecA.Fa.FaCOB -> categorical
'''

'''
Convert the list of children/parents' country of birth to a single variable depicting how many foreigner children/parents the candidate has.
'''

#>
r = re.compile('p1.SecB.Chd.*.ChdCOB')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
feature_name.extend([c for c in data.columns.values if 'MoCOB' in c or 'FaCOB' in c or 'SpsCOB' in c])
display(feature_name)

#>
output_name = 'VisaResult'
display(data.groupby([*feature_name, output_name])[output_name].count())

#>
# replace rows of previous country of residency to count of them
agg_column_name = 'p1.SecB.ChdMoFaSps.X.ChdCOB.ForeignerCount'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_foreigner_family,
                                   agg_column_name=agg_column_name, columns=feature_name)
# delete redundant columns tnx to newly created 'p1.SecB.Chd.X.ChdCOB.ForeignerCount'
data.drop(feature_name, axis=1, inplace=True)

#>
display(data[data[agg_column_name] > 0][agg_column_name])

#>
output_name = 'VisaResult'
feature_name = 'p1.SecB.ChdMoFaSps.X.ChdCOB.ForeignerCount', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(2, 1, figsize=(23, 16))
#data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name].value_counts().plot.pie(
#    autopct='%1.2f%%', ax=ax[0], shadow=True, textprops={'color':'black'})
#ax[0].set_title(output_name)
#ax[0].set_ylabel('')
#
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax[1])
#ax[1].set_title('{} and {}: {}'.format(
#    feature_name[0], feature_name[1], output_name))
#ax[1].legend(loc='upper left')
#for c in ax[1].containers:
#    ax[1].bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
### p1.SecA.App.ChdMStatus -> categorical[/rank?]
'''

'''
TODO: ask agents if we can rank type of marriage and convert this into a **ordered** list?
'''

#>
output_name = 'VisaResult'
feature_name = 'P1.MS.SecA.MS', 'p1.SecA.App.ChdMStatus'

display(data.groupby([*feature_name, output_name])[output_name].count())

'''
There is some inconsistency between two fields, some cases person put `divorced` in one field, and `widowed` in another field. There are cases where second field has value of `9` (unknown), but in other field, filled with some valid value. so after fixing these inconsistencies, this field contains all information about marriage, so we can drop marriage status obtained from 5256e and use 5645e marital statuses.
'''

#>
display(data[data[feature_name[1]] == 9].groupby([*feature_name, output_name])[output_name].count())

#>
# convert numbers to names
ms_name_num = {
    'married': 5,
    'single': 7,
    'common-law': 2,
    'divorced': 3,
    'separated': 4,
    'widowed': 8,
    'annulled': 1,
}
data.loc[data[feature_name[1]] == 9, feature_name[1]] = data[data[feature_name[1]] == 9][feature_name[0]].apply(lambda x: ms_name_num[x])

# status=9 should no longer exist
display(data.groupby([*feature_name, output_name])[output_name].count())

#>
# drop useless
data.drop([feature_name[0]], axis=1, inplace=True)

'''
### p1.SecA.Sps.SpsOcc -> categorical -> continuous
'''

'''
TODO: currently I don't know how convert jobs to continuous or ordered values, so, we just drop em all!
'''

#>
feature_name = [c for c in data.columns.values if (('Occ' in c) and ('OccRow' not in c))]
data.drop(feature_name, axis=1, inplace=True)

'''
### p1.SecB.Chd.X.ChdAccomp -> categorical -> rank
'''

'''
Convert the list of tier 1 family members' accompanying status to a single variable depicting how many of them are accompanying the candidate.
'''

#>
r = re.compile('p1.SecB.Chd.*.ChdAccomp')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# replace rows of children's accompany status to count of them
agg_column_name = 'p1.SecB.Chd.X.ChdAccomp.Count'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_accompanying,
                                   agg_column_name=agg_column_name, columns=feature_name)
# delete redundant columns tnx to newly created 'p1.SecB.Chd.X.ChdAccomp.Count'
data.drop(feature_name, axis=1, inplace=True)

#>
output_name = 'VisaResult'
feature_name = 'p1.SecB.Chd.X.ChdAccomp.Count', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name, feature_name, output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
Following groups have higher chance:
1. `family visit` is generally good except with *large=`3`* family size
2. `family visit` chance decreases by increase in number of f1 family size
3. `tourism` chance increases by increasing the number of accompanying people
4. `visit` only is viable if you are going with `0` accompanying, not good if you going to `visit` with f1 member
'''

'''
### p1.SecA.Mo.MoAccomp, p1.SecA.Fa.FaAccomp -> categorical -> rank
'''

#>
r = re.compile('p1.*.(Fa|Mo)Accomp')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# replace rows of Mo/Fa accompany status to count of them
agg_column_name = 'p1.SecA.ParAccomp.Count'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_accompanying,
                                   agg_column_name=agg_column_name, columns=feature_name)
# delete redundant columns tnx to newly created 'p1.SecB.Chd.X.ChdAccomp.Count'
data.drop(feature_name, axis=1, inplace=True)

#>
output_name = 'VisaResult'
feature_name = 'p1.SecA.ParAccomp.Count', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name, feature_name, output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
### p1.SecA.Sps.SpsAccomp -> categorical -> rank
'''

#>
r = re.compile('p1.SecA.Sps.SpsAccomp')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# replace rows of Spouse accompany status to count of them
agg_column_name = 'p1.SecA.Sps.SpsAccomp.Count'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_accompanying,
                                   agg_column_name=agg_column_name, columns=feature_name)
# delete redundant columns tnx to newly created 'p1.SecB.Chd.X.ChdAccomp.Count'
data.drop(feature_name, axis=1, inplace=True)

#>
output_name = 'VisaResult'
feature_name = 'p1.SecA.Sps.SpsAccomp.Count', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name, feature_name, output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
### p1.SecC.Chd.X.ChdAccomp -> categorical -> rank
'''

'''
Convert the list of tier 1 family members' accompanying status to a single variable depicting how many of them are accompanying the candidate.
'''

#>
r = re.compile('p1.SecC.Chd.*.ChdAccomp')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# replace rows of previous country of residency to count of them
agg_column_name = 'p1.SecC.Chd.X.ChdAccomp.Count'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_accompanying,
                                   agg_column_name=agg_column_name, columns=feature_name)
# delete redundant columns tnx to newly created 'p1.SecC.Chd.X.ChdAccomp.Count'
data.drop(feature_name, axis=1, inplace=True)

#>
output_name = 'VisaResult'
feature_name = 'p1.SecC.Chd.X.ChdAccomp.Count', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name, feature_name, output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
There is not much, except family visit which was already a dominant factor, regardless of the accompanying type and number. 
'''

#>
r = re.compile('.*Accomp')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
feature_name.extend(['P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit'])
display(feature_name)

#>
f, ax = plt.subplots(4, 1, figsize=(22, 18))
z = data[(data[output_name] == 'acc') | (data[output_name] == 'rej')].copy()
z = z[(z[feature_name[-1]] != 'other') & (z[feature_name[-1]] != 'business')]
z[output_name] = z[output_name].apply(lambda x: True if x == 'acc' else False)

for i in range(4):
    sns.pointplot(x=feature_name[i], y=output_name, hue=feature_name[-1],
                data=z, 
                kind='point', ax=ax[i])
plt.show()

'''
Insights:
1. It is clear that bringing any of `siblings` is very risky and decrease chance even for `family visit` which all the time increased the chance!
2. Bringing `spouse` for `visit` purpose is a terrible idea
'''

'''
### p1.SecA.Sps.SpsDOB.Period -> continuous
'''

#>
output_name = 'VisaResult'
feature_name = ['p1.SecA.Sps.SpsDOB.Period', 'p1.SecA.Mo.MoDOB.Period', 'p1.SecA.Fa.FaDOB.Period']
display(data[feature_name].isna().sum())

#>
# convert to years but continuous
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

'''
We have deceased people and their age has been calculated from the day the data has been gathered. I.e. If someone died 30 years before the data was gathered, I still calculate the age like the person is still alive (when a bug makes a lot of copium!). Here, we set max age to 85 (❁´◡`❁).
'''

#>
data[feature_name] = data[feature_name].applymap(lambda x: 85. if x > 85 else x)

#nb>
#display(data[feature_name].describe())
#
#f, ax = plt.subplots(1, 3, figsize=(22, 8))
#for i, c in enumerate(feature_name):
#    sns.histplot(data[c], ax=ax[i], kde=True, stat='percent')
#    ax[i].set_title('{}'.format(c))
#    ax[i].set_xticks(range(0, 90, 7), rotation='vertical', ha='right')
#    ax[i].set_yticks(range(0, 30, 5))
#plt.show()

'''
### p1.SecB.Chd.[X].ChdDOB.Period and p1.SecB.Chd.[X].ChdMStatus and p1.SecB.Chd.[X].ChdRel
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecB.Chd\..*\.(ChdDOB.Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# convert to years but continuous
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

#>
output_name = 'VisaResult'
r = re.compile('p1.SecB.Chd\..*\.(ChdMStatus|ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

'''
I am going to use above features to find "ghost" and "incomplete" kids and fill them out properly. So, none of them should be `None`.
'''

#>
assert data[feature_name].isna().sum().sum() == 0

'''
#### Fill `None`s of p1.SecB.Chd.[X].ChdDOB.Period
'''

#>
r = re.compile('p1.SecB.Chd\..*\.(Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
display(data[list(data.columns.values[mask])].isna().sum())

'''
Note that there are two types of `None`s originally:
1. Ghost children: Where child doest not exist at the first place and we filled `ChdRel = 'OTHER'` and `ChdMStatus = 9`, and `ChdDOB.Period = 0`. Already done when loading data.
2. Incomplete children: Where child exists but `ChdDOB.Period` is `None`. We like to fill this by other children if exists (already done when loading data), but when there is no other child or all children are missing `ChdDOB.Period`, we have to use statistical measure over all dataset to fill it. E.g. average difference of children's age from their parents decreased from missing cases' parent age.
'''

#>
for i in range(len(feature_name) // 2):
    target_feature_name = 'p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period'
    display(data.loc[(data[feature_name[i*2]] == 9) & (data[feature_name[i*2+1]] == 'OTHER') & (data[target_feature_name].isna()), target_feature_name].isna().sum())

'''
We have no instances of case (1) that has not been handled already.
'''

#>
for i in range(len(feature_name) // 2):
    target_feature_name = 'p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period'
    display(data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') & (data[target_feature_name].isna()), target_feature_name].isna().sum())

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = [
    'p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    display(data[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') &
          (data[target_feature_name[i]].isna())][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i]]])

'''
As we can see, case (2) is existence here. Now, I fill it!
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecB.Chd\..*\.(ChdMStatus|ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
target_feature_name = [
    'p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
aggregator_feature_name = 'P1.PD.DOBYear.Period'
for i in range(len(feature_name) // 2):
    parent_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  aggregator_feature_name].median().item()
    child_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  target_feature_name[i]].median().item()
    average_difference = parent_age_median - child_age_median  # must be positive (parent - child) since parent > child
    display('median difference of parents age from "{}"th child: {}'.format(i, average_difference))

    # parent who have children but their children have no age
    na_parent_age = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') &
             (data[target_feature_name[i]].isna()),  aggregator_feature_name]
    fill_age = na_parent_age - average_difference
    fill_age.loc[fill_age < 0.] = 1.  # if fill value become negative, use 1 as the age (too young!)
    # fill the age of "parent who have children but their children have no age"
    data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') &
             (data[target_feature_name[i]].isna()),  target_feature_name[i]] = fill_age

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[38, 256, 273]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

i = 1  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[38, 256]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

'''
So I managed to clap the cheeks of those who didn't fill out their damn children's age.
'''

'''
#### Fill `None`s of p1.SecB.Chd.[X].ChdDOB.Period (case 2 but also rel missing)
'''

#>
r = re.compile('p1.SecB.Chd\..*\.(Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
display(data[list(data.columns.values[mask])].isna().sum())

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = [
    'p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    display(data[(data[target_feature_name[i]].isna())][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i]]])

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[41, 120]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

i = 1  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[41]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

#>
target_feature_name = [
    'p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    aggregator_feature_name = 'P1.PD.DOBYear.Period'
    parent_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  aggregator_feature_name].median().item()
    child_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  target_feature_name[i]].median().item()
    average_difference = parent_age_median - child_age_median  # must be positive (parent - child) since parent > child
    display('median difference of parents age from "{}"th child: {}'.format(i, average_difference))

    # parent who have children but their children have no age and some other features (e.g. rel)
    na_parent_age = data.loc[(data[target_feature_name[i]].isna()),  aggregator_feature_name]
    fill_age = na_parent_age - average_difference
    fill_age.loc[fill_age < 0.] = 1.  # if fill value become negative, use 1 as the age (too young!)
    # fill the age of "parent who have children but their children have no age and some other features (e.g. rel)"
    data.loc[(data[target_feature_name[i]].isna()),  target_feature_name[i]] = fill_age

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[41, 120]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

i = 1  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecB.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[41]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

'''
### p1.SecB.Chd.[X].ChdRel -> categorical
'''

'''
Since there are some `other` in `ChdRel` here that are not "ghost case", we use `ChdDOB.Period` that have been already cleaned and filled based on conditions to detect "ghost" case from "incomplete" case. Hence, in this section, we just rely on non-zero cases of `ChdDOB.Period` to find out if child exists or not!
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecB.Chd\..*\.(ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
for f in feature_name:
    display(data[f].unique())

dic = {
    'DOUGHTER': 'daughter',
    'my daughter': 'daughter',
    'STEPDAUGHTER': 'step daughter',
    'STEP SON': 'step son',
    'SISTER': 'daughter',
    'BOY': 'son',
    'STEP DAUGHTER': 'step daughter',
    'STEP-SON': 'step son',
    'OTHER': 'other'
}

for f in feature_name:
    data[f] = data[f].apply(lambda x: dic[x] if x in dic.keys() else x.lower())

display('--- cleaned ---')
for f in feature_name:
    display(data[f].unique())

#>
output_name = 'VisaResult'
r = re.compile('p1.SecB.Chd\..*\.(ChdDOB.Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# replace rows of previous country of residency to count of them
agg_column_name = 'p1.SecB.Chd.X.ChdRel.ChdCount'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_rel,
                                   agg_column_name=agg_column_name, columns=feature_name)

#nb>
#data[[*feature_name, agg_column_name]]

'''
#### Filling None "ghost case" `other` Randomly
'''

'''
This part is about converting `OTHER` as child-parent relation into `SON` or `DAUGHTER`. Possible methods:
1. Should I even do it? There is no context information to infer and filling it statistically is not rational
2. Biased random fill:
   1. with 50% chance if no kid or all kids are `OTHER`
   2. 75% to be boy if more than half of the kids are *girl*
   3. 25% to be boy if more than half of the kids are *boy*
3. Pure random fill: Just use %50 for any `OTHER` case <- **CURRENT IMPL.** 
'''

'''
I may wanna convert these into `['son', 'daughter', 'other']` or `['son', 'daughter', 'step', 'other']`
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecB.Chd\..*\.(ChdDOB.Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
r = re.compile('p1.SecB.Chd\.\[.*\]\.(ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
target_feature_name = list(data.columns.values[mask])
display(feature_name)
display(target_feature_name)

#>
for f, tf in zip(feature_name, target_feature_name):
    rng = np.random.default_rng(SEED)
    random_child = 'son' if rng.random() > 0.5 else 'daughter'
    data.loc[(data[f] != 0) & (data[tf] == 'other'), tf] = random_child

#nb>
#data[[*feature_name, *target_feature_name]].sample(5, random_state=SEED)

#nb>
#data.sample(1, random_state=SEED)

#>
output_name = 'VisaResult'
feature_name = 'p1.SecB.Chd.X.ChdAccomp.Count', 'p1.SecB.Chd.X.ChdRel.ChdCount'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name, feature_name, output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
Obviously, the children you have and fewer of them are coming with you, the higher chance you have!
'''

'''
### p1.SecC.Chd.[X].ChdDOB.Period and p1.SecC.Chd.[X].ChdMStatus and p1.SecC.Chd.[X].ChdRel
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecC.Chd\..*\.(ChdDOB.Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# convert to years but continuous
data[feature_name] = data[feature_name].apply(lambda x: x/365.)

#>
output_name = 'VisaResult'
r = re.compile('p1.SecC.Chd\..*\.(ChdMStatus|ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

'''
I am going to use above features to find "ghost" and "incomplete" siblings and fill them out properly. So, none of them should be `None`.
'''

#>
assert data[feature_name].isna().sum().sum() == 0

'''
#### Fill `None`s of p1.SecC.Chd.[X].ChdDOB.Period
'''

#>
r = re.compile('p1.SecC.Chd\..*\.(Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
display(data[list(data.columns.values[mask])].isna().sum())

'''
Note that there are two types of `None`s originally:
1. Ghost siblings: Where sibling doest not exist at the first place and we filled `ChdRel = 'OTHER'` and `ChdMStatus = 9`, and `ChdDOB.Period = 0`. Already done when loading data.
2. Incomplete siblings: Where sibling exists but `ChdDOB.Period` is `None`. We like to fill this by other siblings if exist (already done when loading data), but when there is no other sibling or all siblings are missing `ChdDOB.Period`, we have to use statistical measure over all dataset to fill it. E.g. average difference of sibling's age from the main applicant decreased from missing cases' main applicant's age.
'''

#>
# ghost siblings' age must be zero
for i in range(len(feature_name) // 2):
    target_feature_name = 'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period'
    data.loc[(data[feature_name[i*2]] == 9) & (data[feature_name[i*2+1]] == 'OTHER'), target_feature_name] = 0.

#>
for i in range(len(feature_name) // 2):
    target_feature_name = 'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period'
    display(data.loc[(data[feature_name[i*2]] == 9) & (data[feature_name[i*2+1]] == 'OTHER') & (data[target_feature_name].isna()), target_feature_name].isna().sum())

'''
We have no instances of case (1) that has not been handled already.
'''

#>
for i in range(len(feature_name) // 2):
    target_feature_name = 'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period'
    display(data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') & (data[target_feature_name].isna()), target_feature_name].isna().sum())

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = [
    'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    display(data[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') &
          (data[target_feature_name[i]].isna())][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i]]])

'''
As we can see, case (2) is existence here. Now, I fill it!
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecC.Chd\..*\.(ChdMStatus|ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
target_feature_name = [
    'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    aggregator_feature_name = 'P1.PD.DOBYear.Period'
    parent_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  aggregator_feature_name].median().item()
    child_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  target_feature_name[i]].median().item()
    average_difference = parent_age_median - child_age_median  # must be positive (parent - child) since parent > child
    display('median difference of parents age from "{}"th child: {}'.format(i, average_difference))

    # people who have siblings but their siblings have no age
    na_parent_age = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') &
             (data[target_feature_name[i]].isna()),  aggregator_feature_name]
    fill_age = na_parent_age - average_difference
    fill_age.loc[fill_age < 0.] = 1.  # if fill value become negative, use 1 as the age (too young!)
    # fill the age of "people who have siblings but their siblings have no age"
    data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER') &
             (data[target_feature_name[i]].isna()),  target_feature_name[i]] = fill_age

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[85, 166, 273]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

i = 1  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[85, 166, 273]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

i = 2  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[85, 166]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

'''
So I managed to clap the cheeks of those who didn't fill out their damn children's age.
'''

'''
#### Fill `None`s of p1.SecC.Chd.[X].ChdDOB.Period (not case 1 or 2)
'''

#>
r = re.compile('p1.SecC.Chd\..*\.(Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
display(data[list(data.columns.values[mask])].isna().sum())

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = [
    'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    display(data[(data[target_feature_name[i]].isna())][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i]]])

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[129]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

#>
target_feature_name = [
    'p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
for i in range(len(feature_name) // 2):
    aggregator_feature_name = 'P1.PD.DOBYear.Period'
    parent_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  aggregator_feature_name].median().item()
    child_age_median = data.loc[(data[feature_name[i*2]] != 9) & (data[feature_name[i*2+1]] != 'OTHER'),  target_feature_name[i]].median().item()
    average_difference = parent_age_median - child_age_median  # must be positive (parent - child) since parent > child
    display('median difference of parents age from "{}"th child: {}'.format(i, average_difference))

    # people who have siblings but their siblings have no age and some other features (e.g. rel)
    na_parent_age = data.loc[(data[target_feature_name[i]].isna()),  aggregator_feature_name]
    fill_age = na_parent_age - average_difference
    fill_age.loc[fill_age < 0.] = 1.  # if fill value become negative, use 1 as the age (too young!)
    # fill the age of "people who have siblings but their siblings have no age and some other features (e.g. rel)"
    data.loc[(data[target_feature_name[i]].isna()),  target_feature_name[i]] = fill_age

#>
i = 0  # could be 0, 1, 2, 3
target_feature_name = ['p1.SecC.Chd.[' + str(i) + '].ChdDOB.Period' for i in range(len(feature_name) // 2)]
display(data.iloc[[129]][[feature_name[i*2], feature_name[i*2+1], target_feature_name[i], aggregator_feature_name]])

'''
### p1.SecC.Chd.[X].ChdRel -> categorical
'''

'''
Since there are some `other` in `ChdRel` here that are not "ghost case", we use `ChdDOB.Period` that have been already cleaned and filled based on conditions to detect "ghost" case from "incomplete" case. Hence, in this section, we just rely on non-zero cases of `ChdDOB.Period` to find out if sibling exists or not!
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecC.Chd\..*\.(ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
for f in feature_name:
    display(data[f].unique())

typos = {
    'BROTHER': 'brother',
    'SISTER': 'sister',
    'SISTRE': 'sister',
    'SITER': 'sister',
    'NROTHER': 'brother',
    'BRITHER': 'brother',
    'BROTEHR': 'brother',
    'STEP-BROTEHR': 'step brother',
    'BRTHER': 'brother',
    'Siser': 'sister',
    'OTHER': 'other',
}

for f in feature_name:
    data[f] = data[f].apply(functional.fix_typo, args=(typos, None, ))
    data[f] = data[f].apply(str.lower)

display('--- cleaned ---')
for f in feature_name:
    display(data[f].unique())

#>
output_name = 'VisaResult'
r = re.compile('p1.SecC.Chd\..*\.(ChdDOB.Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
# replace rows of previous country of residency to count of them
agg_column_name = 'p1.SecC.Chd.X.ChdRel.ChdCount'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_rel,
                                   agg_column_name=agg_column_name, columns=feature_name)

#nb>
#data[[*feature_name, agg_column_name]]

'''
#### Filling None "ghost case" `other` Randomly
'''

'''
This part is about converting `OTHER` as sibling-main relation into `brother` or `sister`. Possible methods:
1. Should I even do it? There is no context information to infer and filling it statistically is not rational
2. Biased random fill:
   1. with 50% chance if no sibling or all siblings are `other`
   2. 75% to be `brother` if more than half of the siblings are `sister`
   3. 25% to be `brother` if more than half of the siblings are `brother`
3. Pure random fill: Just use %50 for any `other` case <- **CURRENT IMPL.** 
'''

'''
I may wanna convert these into `['brother', 'sister', 'other']` or `['brother', 'sister', 'step', 'other']`
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecC.Chd\..*\.(ChdDOB.Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
r = re.compile('p1.SecC.Chd\.\[.*\]\.(ChdRel)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
target_feature_name = list(data.columns.values[mask])
display(feature_name)
display(target_feature_name)

#nb>
#data[[*feature_name, *target_feature_name]].sample(5, random_state=SEED)

#>
for f, tf in zip(feature_name, target_feature_name):
    rng = np.random.default_rng(SEED)
    random_child = 'brother' if rng.random() > 0.5 else 'sister'
    data.loc[(data[f] != 0) & (data[tf] == 'other'), tf] = random_child

#nb>
#data[[*feature_name, *target_feature_name]].sample(5, random_state=SEED)

#>
output_name = 'VisaResult'
feature_name = 'p1.SecC.Chd.X.ChdRel.ChdCount', 'p1.SecC.Chd.X.ChdAccomp.Count'

#nb>
#ct = pd.crosstab([data[feature_name[0]], data[feature_name[1]]], data[(data[output_name] == 'acc') | (data[output_name] == 'rej')][output_name],
#            margins=True, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name, feature_name, output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

'''
Insights:
1. There is almost no case where someone got visa when going with siblings except very rare cases
2. By increase in number of siblings, the rejection rate decreases considerably. This might be just a correlation behavior since usually people with higher number of siblings (>3 especially) are from traditional families that marry sooner (higher marriage period), are older (zoomers all are alone), have more children, etc. Hence, more likely to go back. This is clearly visible for cases with 5 or more siblings that almost acceptance rate surpasses %50.

'''

'''
### p1.SecA.Fa.ChdMStatus and p1.SecA.Fa.FaDOB.Period
'''

#>
output_name = 'VisaResult'
feature_name = ['p1.SecA.Fa.ChdMStatus', 'p1.SecA.Fa.FaDOB.Period']

#>
display(data[(data[feature_name[0]] == 9) & (data[feature_name[1]] == 0.)].shape[0])

'''
For some weird reason, %10 (28/298) of our applicants have provided zero info about their fathers but it seems irrational to think they don't have a known father, so assuming they haven't provided the info, makes sense.

To fill this, we can follow similar path as children and siblings' age, and just fill `0`s with the average difference of age of mothers from fathers (given social trends about age, `fathers > mothers`)
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.SecA.(Fa|Mo)\..*(ChdMStatus|Period)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
father_feature_name = [v for v in feature_name if 'Fa' in v]
mother_feature_name = [v for v in feature_name if 'Mo' in v]
display(feature_name)
display(father_feature_name)
display(mother_feature_name)

'''
#### Fill Cases With **One** of Mother or Father Age Missing
'''

#>
display(data[(data[father_feature_name[0]] != 9) & (data[father_feature_name[1]] == 0) & (
    data[mother_feature_name[0]] != 9) & (data[mother_feature_name[1]] != 0)][feature_name])
display(data[(data[father_feature_name[0]] != 9) & (data[father_feature_name[1]] != 0) & (
    data[mother_feature_name[0]] != 9) & (data[mother_feature_name[1]] == 0)][feature_name])

#>
father_age_median = data.loc[(data[father_feature_name[0]] != 9) & (data[father_feature_name[1]] != 0),  father_feature_name[1]].median().item()
mother_age_median = data.loc[(data[mother_feature_name[0]] != 9) & (data[mother_feature_name[1]] != 0),  mother_feature_name[1]].median().item()
average_difference = father_age_median - mother_age_median  # mostly positive
display('median difference of fathers age from mothers: {}'.format(average_difference))

# fill fathers with no age with available mothers's age
na_mother_age = data.loc[(data[father_feature_name[0]] != 9) & (data[father_feature_name[1]] == 0) & (data[mother_feature_name[1]] != 0), mother_feature_name[1]]
fill_age = na_mother_age + average_difference
# print(data.loc[(data[father_feature_name[0]] != 9) & (data[father_feature_name[1]] == 0 & (data[mother_feature_name[1]] != 0)),  father_feature_name[1]], fill_age)
data.loc[(data[father_feature_name[0]] != 9) & (data[father_feature_name[1]] == 0 & (data[mother_feature_name[1]] != 0)),  father_feature_name[1]] = fill_age

# fill mothers with no age with available fathers's age
na_father_age = data.loc[(data[mother_feature_name[0]] != 9) & (data[mother_feature_name[1]] == 0) & (data[father_feature_name[1]] != 0), father_feature_name[1]]
fill_age = na_father_age + average_difference
# print(data.loc[(data[mother_feature_name[0]] != 9) & (data[mother_feature_name[1]] == 0) & (data[father_feature_name[1]] != 0),  mother_feature_name[1]], fill_age)
data.loc[(data[mother_feature_name[0]] != 9) & (data[mother_feature_name[1]] == 0) & (data[father_feature_name[1]] != 0),  mother_feature_name[1]] = fill_age

#>
display(data.iloc[[110, 198, 219]][feature_name])

'''
#### Fill Cases With **Both** of Mother or Father Age Missing
'''

'''
These group of people are probably dead!
'''

#>
aggregator_feature_name = 'P1.PD.DOBYear.Period'

#>
display(data[(data[father_feature_name[0]] == 9) & (data[father_feature_name[1]] == 0) & (
    data[mother_feature_name[0]] == 9) & (data[mother_feature_name[1]] == 0)][[*feature_name, aggregator_feature_name]])

#>
data2 = data.copy()

#>
# dead people that have age 
cond = (data[father_feature_name[0]] == 9) & (
    data[father_feature_name[1]] != 0)

# average difference of medians of applicants' age from their parents' age (only for applicants with dead parents)
# since I capped to 85, mean and median are 85 too
father_age_median = data.loc[cond, father_feature_name[1]].median().item()
# since I capped to 85, mean and median are 85 too
mother_age_median = data.loc[cond,  mother_feature_name[1]].median().item()
app_age_median = data.loc[cond, aggregator_feature_name].median().item()

app_father_age_diff = father_age_median - app_age_median
app_mother_age_diff = mother_age_median - app_age_median
# here, it's mostly positive, because the median 
average_difference = father_age_median - mother_age_median
display('median difference of dead fathers age from dead mothers: {}\n \
       median age of dead fathers: {}\nmedian age of dead mothers: {}\n'.format(
    average_difference, father_age_median, mother_age_median))

# apps with both parents dead
cond = (data[father_feature_name[0]] == 9) & (data[father_feature_name[1]] == 0) & (
    data[mother_feature_name[0]] == 9) & (data[mother_feature_name[1]] == 0)
# fill deads' age
na_app_age = data.loc[cond, aggregator_feature_name]
display(na_app_age.shape)
# fill dead fathers' age
data.loc[cond, father_feature_name[1]] = na_app_age + app_father_age_diff
# fill dead mothers' age
data.loc[cond, mother_feature_name[1]] = na_app_age + app_mother_age_diff

#>
index = data2[(data2[father_feature_name[0]] == 9) & (data2[father_feature_name[1]] == 0) & (
    data2[mother_feature_name[0]] == 9) & (data2[mother_feature_name[1]] == 0)][[*feature_name, aggregator_feature_name]].index
display(data.loc[index, [*feature_name, aggregator_feature_name]])

'''
#### Age With No Marital Status
People who have their parents have a valid age field but no martial status dedicated which probably is the indication for dead case which that they should be left as it is. For other cases that people actually forgot to fill, we can follow these approaches:
1. Fill it with their mother's status:
   1. mother widowed (`8`)-> father (dead) -> don't change (`9`)
   2. mother single (`7`) -> father other -> don't change (`9`)
   3. mother married/divorced/all other cases -> father same as mother
2. Leave it! **<- CURRENT IMPL.**
'''

#>
display(data[(data[feature_name[0]] == 9) & (data[feature_name[1]] != 0.)].shape[0])

#nb>
#data[(data[father_feature_name[0]] == 9) & (data[father_feature_name[1]] != 0) & (
#    data[mother_feature_name[0]] == 9) & (data[mother_feature_name[1]] != 0)][[*feature_name]]

'''
57 out of 62 cases have dead father and dead mother. So, it means that almost all cases where marital status is missed, probably person is dead and filling it with other values might not seem appropriate.

TODO: until occupation fields are incorporated which contain `deceased` for people who are actually dead.
'''

'''
### p1.SecX.*.*Addr -> int (count)
'''

'''
Since there are some `other` in `ChdRel` here that are not "ghost case", we use `ChdDOB.Period` that have been already cleaned and filled based on conditions to detect "ghost" case from "incomplete" case. Hence, in this section, we just rely on non-zero cases of `ChdDOB.Period` to find out if child exists or not!
'''

#>
output_name = 'VisaResult'
r = re.compile('p1.Sec(A|B|C)\..*\..*(Addr)')
mask = np.isin(data.columns.values, list(filter(r.match, data.columns.values)))
feature_name = list(data.columns.values[mask])
display(feature_name)

#>
data[[*feature_name, output_name]].sample(2, random_state=SEED)

#>
IRAN_PROVINCES = {
    'Alborz': 'Alborz',
    'AKBORZ': 'Alborz',  # Typo case: Alborz
    'Ardabil': 'Ardabil',
    'Ardebil': 'Ardebil',
    'Azarbayjan-e Gharbi': 'West',
    'West Azerbaijan': 'West',
    'Azerbaijan-e Gharbi': 'West',
    'Azerbaijan, West': 'West',
    'West': 'West',  # we only have one province with 'West' in it
    'Gharbi': 'West', # we only have one province with 'Gharbi' in it
    'Azerbaijan-e Sharqi': 'East',
    'Azarbayjan-e Sharqi': 'East',
    'Azerbaijan, East': 'East',
    'East Azerbaijan': 'East',
    'East': 'East',  # we only have one province with 'East' in it
    'Sharqi': 'East', # we only have one province with 'Sharghi' in it
    'Sharghi': 'East', # we only have one province with 'Sharghi' in it
    'Bushehr': 'Bushehr',
    'Chaharmahal and Bakhtiari': 'Chaharmahal and Bakhtiari',
    'Chahar Mahaal and Bakhtiari': 'Chaharmahal and Bakhtiari',
    'Chahar Mahal-e Bakhtiari': 'Chaharmahal and Bakhtiari',
    'Fars': 'Fars',
    'Gilan': 'Gilan',
    'Guilan': 'Gilan',
    'Golestan': 'Golestan',
    'Hamadan': 'Hamadan',
    'Hamedan': 'Hamadan',
    'Hormozgan': 'Hormozgan',
    'Ilam': 'Ilam',
    'Isfahan': 'Isfahan',
    'Esfahan': 'Isfahan',
    'ESFSHAN': 'Isfahan',  # typo case: Esfahan
    'Kerman': 'Kerman',
    'Kermanshah': 'Kermanshah',
    'Khorasan-e Janubi': 'South',
    'Khorasan-e Jonubi': 'South',
    'Khorasan Jonubi': 'South',
    'South Khorasan': 'South',
    'Khorasan, South': 'South',
    'Khorasan Janubi': 'South',
    'Jonubi': 'South',  # we only have one province with 'Jonubi' in it
    'Janubi': 'South', # we only have one province with 'Janubi' in it
    'South': 'South', # we only have one province with 'South' in it
    'Khorasan-e Razavi': 'Razavi',
    'Khorasan, Razavi': 'Razavi',
    'Razavi Khorasan': 'Razavi',
    'Razavi': 'Razavi', # we only have one province with 'Razavi' in it
    'Khorasan-e Shemali': 'North',
    'Khorasan Shomali': 'North',
    'Khorasan-e Shomali': 'North',
    'Khorasan, North': 'North',
    'North Khorasan': 'North',
    'Shomali': 'North',  # we only have one province with 'Shomali' in it
    'Shemali': 'North', # we only have one province with 'Shemali' in it
    'North': 'North', # we only have one province with 'North' in it
    'Khuzestan': 'Khuzestan',
    'KHOUZESTAN': 'Khuzestan',
    'Kohgiluyeh and Buyer Ahmad': 'Kohgiluyeh and Buyer Ahmad',
    'Kohgiluyeh and Boyer-Ahmad': 'Kohgiluyeh and Buyer Ahmad',
    'Kurdistan': 'Kurdistan',
    'Kordestan': 'Kurdistan',
    'Lorestan': 'Lorestan',
    'Markazi': 'Markazi',
    'Mazandaran': 'Mazandaran',
    'Qazvin': 'Qazvin',
    'GHAZVIN': 'Qazvin',
    'Qom': 'Qom',
    'Semnan': 'Semnan',
    'Sistan and Baluchestan': 'Sistan and Baluchestan',
    'Tehran': 'Tehran',
    'Tehra': 'Tehran',  # typo case: Tehran
    'Teh': 'Tehran',  # typo case: Tehran
    'Te': 'Tehran',  # typo case: Tehran
    'TRHRAN': 'Tehran', # typo case: Tehran
    'TEHRTAN': 'Tehran', # typo case: Tehran
    'TEHRANIRAN': 'Tehran', # typo case: Tehran
    'EHRAN': 'Tehran', # typo case: Tehran
    'Urmia': 'Urmia',
    'Orumiyeh': 'Urmia',
    'Orumiye': 'Urmia',
    'Orumieh': 'Urmia',
    'Orumie': 'Urmia',
    'OROMIEH': 'Urmia', # typo case: Urmia
    'Yazd': 'Yazd',
    'Zanjan': 'Zanjan',
}

# hardcoded af
FOREIGN_COUNTRY_MISSED = [
    'USA',
    'UAE',
    'TURKET',
    'Meadow',
    'US',
    'ONTARIO',
    'DUBAI',
    'TORONTO',
    'FRAMINGHAM',
    'SCOTTSDDIE',
    'EMIRATES',
    'OAKLAND',
    'Stockholm',
    'frankfurt',
    'VANCOUVER',
    'TORENTO',
    'JACKSON',
    'ISTANBUL',
    'BRECKENRIDGE',
    'London',

]

from vizard.configs import IRAN_PROVINCE_TO_CITY
from typing import Optional

def make_dictionary_lowercase(dictionary: dict) -> dict:
    return {key.lower(): value.lower() for key, value in dictionary.items()}

IRAN_PROVINCE_TO_CITY_DICT =  make_dictionary_lowercase(functional.config_csv_to_dict(IRAN_PROVINCE_TO_CITY))
IRAN_PROVINCES = make_dictionary_lowercase(IRAN_PROVINCES)
# combine list of countries and manually extracted list of them
country_list = list(functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME).values())
FOREIGN_COUNTRY_MISSED = [country.lower() for country in FOREIGN_COUNTRY_MISSED]
country_list.extend(FOREIGN_COUNTRY_MISSED)

def address_to_city(address: Optional[str],
                    city_province_dict: dict,
                    province_dict: dict,
                    country_list: list) -> Optional[str]:
    """Takes an full address and extracts the city then province containing the city.

    Note:
        Since the goal of defining this function is to aggregate over addresses, if
            the address is not provided, it is valid to ignore it since it cannot be
            inferred with %100 accuracy and any statistical method would be biased
            towards the majority class which is the applicant's address.

    TODO: implement a spell checker to correct the city names.
        You must be aware that the city names also have multiple ways of being spelled
            but we ignore it as it would be too much effort to correct it. 
        Just a reminder, this has been done manually for the provinces and has been
            provided using `province_dict`.
            

    Args:
        address (str, optional): full address which is expected to contain
            the city and province.
        city_province_dict (dict): dictionary mapping city names to provinces
        province_dict (dict): dict of provinces and their typos. This dict
            needs to be separately provided to include different spelling 
            for the name of the same province. This would prevent unnecessary
            duplication of entries in keys of `city_province_dict`.
            Keys are all type of spelling and values are the correct spelling. E.g.::
                {
                    'Kordestan': 'Kurdistan',
                    'Kurdistan': 'Kurdistan'
                }

        country_list (list): list of countries. We consider any address that has an
            instance in this list as 'foreign' to just separate it from "long distance"
            cases where cases are inside a country (here, Iran).

    Returns:
        str: Province of the address
    """

    if address is None:
        return None
    
    # lower case items in country_list
    country_list = [item.lower() for item in country_list]

    addr_part = re.split(r'(,|-|:| )', address)
    # look for city in the address
    for part in addr_part:
        part = part.lower().strip()
        # skip dead cases
        if ('deceased' in part) or ('passed' in part):
            return 'deceased'
        # return province if it is already in address
        if part in province_dict.keys():
            return province_dict[part]
        # return foreign if it is in country list except 'Iran'
        if (part in country_list) and (part != 'iran'):
            return 'foreign'
        # get the province of the city if province not in address
        if part in city_province_dict.keys():
            return city_province_dict[part]
    # if we cannot infer address, just ignore it :D
    print(ValueError(f'Cannot infer city from address: {address}'))
    return None

for f in feature_name:
    data[f] = data[f].apply(address_to_city, args=(IRAN_PROVINCE_TO_CITY_DICT,
                                                   IRAN_PROVINCES,
                                                   country_list, ))


#>
data[[*feature_name, output_name]].sample(5, random_state=SEED)

#>
# count number of family members that are living in different place than applicant (non-foreign)
agg_column_name = 'p1.SecX.LongDistAddr'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_long_distance_family_resident,
                                   agg_column_name=agg_column_name, columns=feature_name)

#>
data[[agg_column_name, *feature_name, output_name]].sample(5, random_state=SEED)

#>
# count number of family members that are living in a foreign country
agg_column_name = 'p1.SecX.ForeignAddr'
canada_logic.reset_dataframe(dataframe=data)
data = canada_logic.add_agg_column(aggregator=canada_logic.count_foreign_family_resident,
                                   agg_column_name=agg_column_name, columns=feature_name)

#>
data[[agg_column_name, *feature_name, output_name]].sample(5, random_state=SEED)

#nb>
#agg_column_name = ['p1.SecX.LongDistAddr', 'p1.SecX.ForeignAddr']
#data[[*feature_name, *agg_column_name]]

#>
# drop aggregated address columns
data.drop(columns=feature_name, axis=1, inplace=True)

'''
#### Vis
'''

#>
output_name = 'VisaResult'
feature_name = ['p1.SecX.LongDistAddr', 'p1.SecX.ForeignAddr', 'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit']

#nb>
#z = data[(data[output_name] == 'acc') | (data[output_name] == 'rej')]
#z = z[(z[feature_name[0]] <= 7) & (z[feature_name[1]] <= 4)]
#ct = pd.crosstab([z[feature_name[0]], z[feature_name[1]]], z[output_name],
#            margins=False, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
## ct.drop(['no idea', 'w-acc', 'w-rej'], axis=1, inplace=True)
#ct

#nb>
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name[0], feature_name[1], output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

#>
f, ax = plt.subplots(2, 1, figsize=(22, 12))
z = data[(data[output_name] == 'acc') | (data[output_name] == 'rej')].copy()
z = z[(z[feature_name[2]] != 'business') & (z[feature_name[2]] != 'other')]
z[output_name] = z[output_name].apply(lambda x: True if x == 'acc' else False)
sns.pointplot(x=feature_name[1], y=output_name, hue=feature_name[2],
              data=z,
              kind='point', ax=ax[0])

sns.pointplot(x=feature_name[0], y=output_name, hue=feature_name[2],
              data=z,
              kind='point', ax=ax[1])
plt.show()

'''
Insights:
1. When `ForeignAddr` increases, the change of getting visa with purpose of `f2` decrease hugely.
2. As expected, visiting `f1` is dominant factor and does not seem that `Addr` have any effect on it.
3. This is the data for all people, single, married, old, young, etc. So a subset of these might provide more info
   1. Single people for `longDistAddr` and purpose `relationship`
'''

'''
##### Single people
'''

#nb>
#z = data[(data[output_name] == 'acc') | (data[output_name] == 'rej')]
#z = z[(z[feature_name[0]] <= 7) & (z[feature_name[1]] <= 4)]
## single people
#z = z[z['p1.SecA.App.ChdMStatus'] == 7]
#
#
#ct = pd.crosstab([z[feature_name[0]], z[feature_name[1]]], z[output_name],
#            margins=False, dropna=False, normalize=True) * 100
#ct.drop(ct.tail(1).index, inplace=True)
#f, ax = plt.subplots(1, 1, figsize=(22, 12))
#ct.plot(kind='bar', stacked=True, rot=90, ax=ax)
#ax.set_title('{} and {}: {}'.format(
#    feature_name[0], feature_name[1], output_name))
#ax.legend(loc='upper right')
#for c in ax.containers:
#    ax.bar_label(c, label_type='center', fmt='%1d%%', color='black')
#plt.show()

#>
f, ax = plt.subplots(2, 1, figsize=(22, 12))
z = data[(data[output_name] == 'acc') | (data[output_name] == 'rej')].copy()
z = z[(z[feature_name[2]] != 'ukn') & (z[feature_name[2]] != 'work')]
z[output_name] = z[output_name].apply(lambda x: True if x == 'acc' else False)

# single people
z = z[z['p1.SecA.App.ChdMStatus'] == 7]

sns.pointplot(x=feature_name[1], y=output_name, hue=feature_name[2],
              data=z,
              kind='point', ax=ax[0])

sns.pointplot(x=feature_name[0], y=output_name, hue=feature_name[2],
              data=z,
              kind='point', ax=ax[1])
plt.show()

'''
No information here.
'''

#nb>
#data.sample(5, random_state=SEED)

#nb>
#data.describe()

'''
## Optimized Datatype

1. **Preference**: I have my own preferences in some valid data types, i.e. `float32` instead of `float64`
   1. continuous data represented as `floatx` to `float32`

2. **Validate**: Some data types are not valid and cannot be used, so it has to be converted:
    1. categorical data represented as `integer` to Pandas `category`
    2. categorical data represented as `string` to Pandas `category`

'''

#>
display(data.info())

#>
unique_counts = pd.DataFrame.from_records([(col, data[col].nunique()) for col in data.columns],
                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
display(unique_counts.transpose())

'''
### To `category`
'''

#>
feature_name = [
    'P1.PD.AliasName.AliasNameIndicator.AliasNameIndicator', 'P3.BGI2.PrevApply', 'P3.refuseDeport', 'P3.noAuthStay', 'P3.Edu.EduIndicator', 
    'P2.MS.SecA.PrevMarrIndicator', 'P1.PD.SameAsCORIndicator', 'P1.PD.Sex.Sex', 'p1.SecB.Chd.[2].ChdMStatus', 'p1.SecB.Chd.[3].ChdMStatus', 
    'p1.SecC.Chd.[2].ChdRel', 'p1.SecC.Chd.[3].ChdRel', 'p1.SecC.Chd.[4].ChdMStatus', 'p1.SecB.Chd.[1].ChdMStatus', 'p1.SecC.Chd.[5].ChdRel',
    'p1.SecC.Chd.[6].ChdRel', 'p1.SecC.Chd.[4].ChdRel', 'p1.SecB.Chd.[1].ChdRel', 'P1.PD.CWA.Row2.Status', 'p1.SecC.Chd.[5].ChdMStatus', 'p1.SecC.Chd.[1].ChdRel',
    'p1.SecC.Chd.[0].ChdRel', 'p1.SecB.Chd.[3].ChdRel', 'p1.SecC.Chd.[3].ChdMStatus', 'p1.SecC.Chd.[2].ChdMStatus', 'p1.SecC.Chd.[1].ChdMStatus', 
    'p1.SecB.Chd.[2].ChdRel', 'P3.cntcts_Row2.Relationship.RelationshipToMe', 'p1.SecB.Chd.[0].ChdMStatus', 'p1.SecA.App.ChdMStatus', 'p1.SecC.Chd.[6].ChdMStatus',
    'P3.DOV.PrpsRow1.PrpsOfVisit.PrpsOfVisit', 'VisaResult', 'p1.SecB.Chd.[0].ChdRel', 'P3.DOV.cntcts_Row1.RelationshipToMe.RelationshipToMe',
    'p1.SecA.Fa.ChdMStatus', 'p1.SecA.Mo.ChdMStatus', 'p1.SecC.Chd.[0].ChdMStatus', 'P1.PD.CurrCOR.Row2.Status', 'P1.PD.CWA.Row2.Country',
    'P3.Edu.Edu_Row1.FieldOfStudy', 'P3.Occ.OccRow1.Occ.Occ', 'P3.Occ.OccRow2.Occ.Occ', 'P3.Occ.OccRow3.Occ.Occ'
]

#nb>
#data[feature_name].info()

'''
#### convert all those containing 'Status' to `int` from `float64`
'''

#>
feature_name_status = [c for c in feature_name if 'Status' in c]
# float to int since status is categorical
data[feature_name_status] = data[feature_name_status].astype(int)
# int to categorical
data[feature_name_status] = data[feature_name_status].astype('category')

'''
#### convert `string` to `category`
'''

#>
for c in feature_name:  # all have to be type of `object`
    if (data[c].dtype == 'string'):
        display(c)
    data[c] = data[c].astype('category')

'''
#### convert `object` to `category`
'''

#>
# NBVAL_IGNORE_OUTPUT
feature_name_obj = list(set(feature_name) - set(feature_name_status))
for c in feature_name_obj:  # all have to be type of `object`
    assert (data[c].dtype == 'object') or (data[c].dtype == 'category')

data[feature_name_obj] = data[feature_name_obj].astype('category')
display(feature_name_obj)

#nb>
#data[feature_name].info()

'''
#### To `32` bit (`float` and `int`)
'''

#>
feature_name_continuous = list(set(data.columns.values) - set(feature_name))

#nb>
## NBVAL_IGNORE_OUTPUT
#data[feature_name_continuous].info()

#>
for c in feature_name_continuous:
    if data[c].dtype == 'int64':
        data[c] = data[c].astype(np.int32)
    if data[c].dtype == 'float64':
        data[c] = data[c].astype(np.float32)

#nb>
## NBVAL_IGNORE_OUTPUT
#data[feature_name_continuous].info()

#>
display(data.info())

'''
### Correlation Between The Features
'''

#nb>
#
## ref https://datascience.stackexchange.com/a/82237/135109
#from itertools import combinations
#
#
#def abs_high_pass(df, abs_thresh):
#    passed = set()
#    for (r, c) in combinations(df.columns, 2):
#        if (abs(df.loc[r, c]) >= abs_thresh):
#            passed.add(r)
#            passed.add(c)
#    passed = sorted(passed)
#    return df.loc[passed, passed]
#
#corr_data = data.corr()
#sns.heatmap(abs_high_pass(corr_data, 0.5), annot=True, cmap='RdYlGn',
#            linewidths=0.2, xticklabels=True, yticklabels=True)
#fig = plt.gcf()
#fig.set_size_inches(22, 22)
#plt.show()

'''
Highly correlated data:
1. DateOfMarr 
   1. with all other "age" related periods, and the later the child/sibling, the lower the correlation (later children are younger)
   2. Chd.Count since the older the person the more chance of having more children
2. OccRow1.Country and OccRow2.Country since most of the people have the same Country
3. PrevCOR.Row3.Country and PrevCOR.Row3.Period have max correlation since its the exact same for all (all 0)
   1. practically we can drop both of them! (theoretically dropping one should be done)
4. P3.Edu.Edu_Row1.Period has *negative* correlation with marriage and children factors. I.e. the more education person has, the higher age of marriage and fewer children which are having them in older ages
5. P3.Edu.Edu_Row1.Country: same as (4) with this difference that the correlation is *twice negative* as (4). I.e. if person studies in a country better than default (here Iran), it's way less likely to get married.
6. p1.SecB.ChdX.ChdDOB.Period all have positive correlation with each other and factors in (1) since the more traditional, the younger the marriage age, the more kids, more siblings, lower education, higher marriage period, and so on.
'''

'''
### Dimensionality Reduction for Visualization
1. PCA 
2. t-SNE
'''

#nb>
#data_encoded = data.copy()
#columns_to_encode = [c for c in data.columns.values if (data[c].dtype == 'category') and (c != 'VisaResult')]
#data_encoded = pd.get_dummies(data_encoded, drop_first=True, columns=columns_to_encode)
#data_encoded.shape

#nb>
#blah = [c for c in data_encoded.columns.values if data_encoded[c].isna().sum() > 0]
#display(blah)
#data_encoded[blah] = data_encoded[blah].fillna(0)

#nb>
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#from sklearn.preprocessing import LabelEncoder
#
#Y = data_encoded['VisaResult'].to_numpy()
#y_label_encoder = LabelEncoder()
#Y = y_label_encoder.fit_transform(Y)
#data_encoded.drop(['VisaResult'], axis=1, inplace=True)
#X = data_encoded.to_numpy()

'''
#### PCA
'''

#nb>
#pca = PCA(n_components=2)
#Xt = pca.fit_transform(X)
#
#display(Xt.shape)
#display(Y.shape)

#nb>
#plt.figure(figsize=(22, 12))
#plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=Y.flatten())
#plt.legend(handles=plot.legend_elements()[0], labels=list(y_label_encoder.classes_))
#plt.show()

#nb>
#pca = PCA(n_components=3)
#Xt = pca.fit_transform(X)
#
#display(Xt.shape)
#display(Y.shape)

#nb>
#fig = plt.figure(figsize=(22, 12))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=Y.flatten())
#ax.legend(handles=plot.legend_elements()[0], labels=list(y_label_encoder.classes_))
#plt.show()

'''
#### t-SNE
'''

#nb>
#tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', method='exact', init='pca', random_state=SEED, n_jobs=-1)
#Xt = tsne.fit_transform(X)
#
#display(Xt.shape)
#display(Y.shape)

#nb>
#plt.figure(figsize=(22, 12))
#plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=Y.flatten())
#plt.legend(handles=plot.legend_elements()[0], labels=list(y_label_encoder.classes_))
#plt.show()

#nb>
#tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', method='exact', init='pca', random_state=SEED, n_jobs=-1)
#Xt = tsne.fit_transform(X)
#
#display(Xt.shape)
#display(Y.shape)

#nb>
#fig = plt.figure(figsize=(22, 12))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=Y.flatten())
#ax.legend(handles=plot.legend_elements()[0], labels=list(y_label_encoder.classes_))
#plt.show()

'''
## Save Data to Disc
**Save preprocessed data to disc, as a baseline. Why:**
1. All type casting, feature engineering, visualization, etc all needed to be added to the main package to ensure reproducibility as being part of the same end to end pipeline
2. During the transition happening at step (1), we check the correct implementation by comparing to this saved data
3. I am tired af of EDA and all software engineering and cheeky if else pleb approaches of data analysis, doing some new stuff like weak supervision for converting non/weak labeled data to absolute one would add some fun
4. It's not gonna cost anything, so why not!


**Tracking:**
1. This data need to be tracked by DVC too
2. Consider this data, `dev` version with a new unique tag (i.e. if last tag is `v1.0.2`, we use `v1.1.0-dev`) and use following format for commit message `EDA of [previous_production_tag_version]`, i.e. `EDA of v1.0.2`
3. After all implementation in this notebook have been merged into original codebase cleanly, we rerun data generation pipeline and create the production ready data which will be tracked and tagged with `v1.1.0` (same as step 2 but without `-dev`)

The reason for this is the step 3 of previous part and we want to track the snorkel procedure too.
'''

#>
# dataset_path = REPO + '/raw-dataset/all-dev.pkl'  # Don't change the name, 
# data.to_pickle(dataset_path)

'''
## Load Data For Comparison
'''

#>
from pandas import testing as tm

DATASET_PATH_ORIGINAL = 'raw-dataset/all-dev.pkl'  # Don't change the name, 
REPO = '/home/nik/visaland-visa-form-utility'
VERSION = 'v1.2.2-dev'  # use latest `dev` version using `git tag`
data_url_original = dvc.api.get_url(path=DATASET_PATH_ORIGINAL, repo=REPO, rev=VERSION)
data_original = pd.read_pickle(data_url_original)

#>
# tm.assert_frame_equal(data, data_original, check_exact=False, rtol=1e-4, atol=1e-5)
import pandas.testing as pdt
BLAH = False
for c in data_original.columns.values:
    try:
        pdt.assert_series_equal(left=data[c], right=data_original[c],
                                check_exact=False, rtol=1e-3, atol=1e-4)
    except AssertionError as e:
        print(f'column="{c}" {e}\n\n')
        BLAH = True
if BLAH:
    raise AssertionError('data not equal')

#nb>
#data

#nb>
## copy notebook to `artifacts` dir to be logged by mlflow
#notebook_name = 'notebooks/data_exploration_dev.ipynb'
#shutil.copy2(notebook_name, REPO + '/artifacts/' + notebook_name)

#nb>
## NBVAL_IGNORE_OUTPUT
#!jupyter nbconvert --to html $notebook_name
#notebook_name_html = notebook_name.replace('ipynb', 'html')
#shutil.copy2(notebook_name_html, REPO + '/artifacts/' + notebook_name_html)
#import pathlib
#pathlib.Path(notebook_name_html).unlink()

'''
## Save MLflow Tracking to Disc
'''

#nb>
## log data params
#mlflow.log_param('data_url', data_url)
#mlflow.log_param('data_url_DEV', data_url_original)
#mlflow.log_param('raw_dataset_dir', DST_DIR)
#mlflow.log_param('EDA_passed_dataset_dir_x-dev', DATASET_PATH_ORIGINAL)
#mlflow.log_param('data_version_after_EDA_vxxx-dev', VERSION)
#mlflow.log_param('helper_dataset_info', all_helper_data_info)
#mlflow.log_param('input_shape', data.shape)
#mlflow.log_param('input_columns', data.columns.values)
#mlflow.log_param('input_dtypes', data.dtypes.values)
#
## Log artifacts (logs, saved files, etc)
#mlflow.log_artifacts(REPO + '/artifacts/')
## delete redundant logs, files that are logged as artifact
#shutil.rmtree(REPO + '/artifacts')
#
## terminate mlflow tracker
#client.set_terminated(mlflow.active_run().info.run_id, status='FINISHED')
