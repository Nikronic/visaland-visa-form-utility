from typing import List
import logging

import pandas as pd
from snorkel.augmentation import TransformationFunction
from snorkel.augmentation import PandasTFApplier
from snorkel.augmentation import ApplyEachPolicy

from vizard_snorkel import augmentation


logger = logging.getLogger(__name__)


def preview_tfs(dataframe: pd.DataFrame, tfs: List[TransformationFunction],
                n_samples: int = 1) -> pd.DataFrame:
    """
    Shows transformed column in a dataframe along side its original value given the list of 
        `TransformationFunction`s for all transformations provided by applying each of
        them individually (i.e. `ApplyEachPolicy`)

    args:
        dataframe: The pandas dataframe that `tfs` can be applied on
        tfs: a list of `snorkel.augmentation.tf.TransformationFunction` instances
    """
    augmentation.series_noise_utils.set_dataframe(df=dataframe)
    columns = ['TF name', 'Original', 'Transformed']

    # apply TFs on sampled dataframe
    policy = ApplyEachPolicy(n_tfs=len(tfs), keep_original=False)
    tf_applier = PandasTFApplier(tfs, policy)

    # logging
    logger.info(f'Showing TF effects for "{n_samples}" samples\
        \n\tusing policy="{policy.__class__.__name__}" for following transformation functions:\
            \n\t{f"{chr(10)}{chr(9)}".join([tf.name for tf in tfs])}')

    result = {}
    samples = dataframe.sample(n_samples)
    samples_augmented = tf_applier.apply(samples)
    for i in range(len(samples)):
        for tf in tfs:
            column_name = tf._resources['column']
            result[f'sample_{i}_{tf.name}'] = [
                samples.iloc[i, :][column_name], samples_augmented.iloc[i, :][column_name]]

    # proper column name
    result = pd.DataFrame.from_dict(
        data=result, orient='index', columns=columns[1:])
    result.rename({'index': columns[0]}, inplace=True)  # type: ignore
    return result
