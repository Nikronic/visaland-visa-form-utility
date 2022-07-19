# core
import numpy as np
import pandas as pd
# ours: models
from vizard.models.preprocessors import ColumnTransformer

# helpers
from typing import List, Union
import logging


# configure logging
logger = logging.getLogger(__name__)


def preview_column_transformer(column_transformer: ColumnTransformer,
                               original: np.ndarray,
                               transformed: np.ndarray,
                               columns: List[str],
                               random_state: Union[int, np.random.Generator] = np.random.default_rng(),
                               **kwargs) -> pd.DataFrame:
    """Preview transformed data next to original one obtained via ``ColumnTransformer``

    TODO:
        - add support for one hot encoding

    Args:
        column_transformer (ColumnTransformer): An instance
            of :class:`sklearn.compose.ColumnTransformer`
        original (np.ndarray): Original data as a :class:`numpy.ndarray`.
            Same shape as ``transformed``
        transformed (np.ndarray): Transformed data as a :class:`numpy.ndarray`.
            Same shape as ``original``
        columns (List[str]): List of column names in original dataframe that
            ``original`` and ``transformed`` are extracted from
        random_state (Union[int, np.random.Generator], optional): A seed value or
            instance of  :class:`numpy.random.Generator` for sampling. Defaults to
            :func:`numpy.random.default_rng()`.
        **kwargs: Additional arguments as follows:

            * ``n_samples`` (int): Number of samples to draw. Defaults to 1.
    
    Raises:
        ValueError: If ``original`` and ``transformed`` are not of the same shape

    Yields:
        pd.DataFrame: Preview dataframe for each transformer in ``column_transformer.transformers_``.
            Dataframe has twice as columns as ``original`` and ``transformed``, i.e.
            ``df.shape == (original.shape[0], 2 * original.shape[1])``
    """
    # extract kwargs
    n_samples = kwargs.get('n_samples', 1)

    # just aliases for shorter lines
    ct = column_transformer
    
    # verify shapes
    if original.shape != transformed.shape:
        raise ValueError('original and transformed must have the same shape')

    # set rng
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    # generate sample indices
    sample_indices = random_state.choice(original.shape[0],
                                         size=n_samples,
                                         replace=False)
    sample_indices = sample_indices.reshape(-1, 1)  # to broadcast properly

    # loop through each transform (over subset of columns) and preview it    
    for idx, k in enumerate(ct.output_indices_):
        # 'remainder' is not transformed, so end of the loop
        if k == 'remainder':
            return None
        # get indices of the transformed columns
        transformed_columns_indices = ct.output_indices_[k]
        # get indices of the original columns
        columns_indices = ct._columns[idx]
        # get names of the original columns
        columns_indices_names = columns[columns_indices]
        # compare the values of the transformed and the original columns
        original_sample = original[sample_indices, columns_indices]
        transformed_sample = transformed[sample_indices, transformed_columns_indices]
        # fix shapes to be 2d
        original_sample = original_sample.reshape(sample_indices.shape[0],
                                                  columns_indices.__len__())
        transformed_sample = transformed_sample.reshape(sample_indices.shape[0],
                                                  columns_indices.__len__())
        # create a dataframe with the original and transformed columns side by side
        sample = np.empty(
            shape=(original_sample.shape[0], original_sample.shape[1] * 2))
        sample[:, ::2] = original_sample
        sample[:, 1::2] = transformed_sample
        preview_cols: List[str] = []
        [preview_cols.extend([f'{c}_og', f'{c}_tf'])  # type: ignore
         for c in columns_indices_names]
        preview_df = pd.DataFrame(sample, columns=preview_cols)
        # yield the previews
        if n_samples == 1:  # just better visuals for single sample
            yield preview_df.T
        else:
            yield preview_df
