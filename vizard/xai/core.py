from typing import Dict, Tuple
from vizard.data.constant import FEATURE_CATEGORY_TO_FEATURE_NAME_MAP, FeatureCategories
import numpy as np


def get_top_k_idx(sample: np.ndarray, k: int) -> np.ndarray:
    """Extracts top-k indices of an array

    Args:
        sample (:class:`numpy.ndarray`): A single instance :class:`numpy.ndarray`
        k (int): Number of items to return

    Return:
        :class:`numpy.ndarray`: List of top-k indices
    """
    top_k_idx: np.ndarray = np.argpartition(sample, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(sample[top_k_idx])][::-1]
    return top_k_idx


def get_top_k(
    sample: np.ndarray, k: int, absolute: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts top-k items in an numpy array conditioned on sign of values

    Args:
        sample (:class:`numpy.ndarray`): A :class:`numpy.ndarray` array
        k (int): Number of items to return
        absolute (bool, optional): Wether or not to consider
            the sign of values in computing top-k. If ``True``,
            then absolute values used and vice versa. Defaults to True.

    Raises:
        NotImplementedError: If ``sample`` contains multiple samples (rows).

    Returns:
        Tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`]:
            Top-k values of array ``sample`` and their indices in a tuple.
    """
    # if single instance is given
    if (sample.ndim == 1) or ((sample.ndim == 2) and (sample.shape[0] == 1)):
        sample = sample.flatten()

        # if absolute top k should be chosen
        top_k_idx: np.ndarray
        if absolute:
            # absolute top k
            sample_abs: np.ndarray = np.abs(sample)
            top_k_idx = get_top_k_idx(sample=sample_abs, k=k)
        else:
            # signed top k
            top_k_idx = get_top_k_idx(sample=sample, k=k)

        # top k values with their signs
        top_k: np.ndarray = sample[top_k_idx]
        return (top_k, top_k_idx)
    else:
        raise NotImplementedError(
            "This method is only available for" " single-instance numpy arrays. (yet)"
        )


def xai_threshold_to_text(xai_value: float, threshold: float = 0.0) -> str:
    """Converts a XAI value to a negative/positive text by thresholding

    Args:
        xai_value (float): XAI value to be interpreted
        threshold (float): XAI threshold

    Returns:
        str: a negative/positive string satisfying ``threshold``
    """

    if xai_value >= threshold:
        return "خوب است"
    else:
        return "بد است"


def xai_to_text(
    xai_feature_values: Dict[str, float], feature_to_keyword_mapping: Dict[str, str]
) -> Dict[str, Tuple[float, str]]:
    """Takes XAI values for features and generates basic textual descriptions

    XAI values are computed using `SHAP` (see :class:`vizard.xai.shap`). Then, I
    use a simple dictionary that has a basic general statement for each feature.
    Via a simple thresholding, I generate textual descriptions for each feature.
    e.g., I take ``"P3.DOV.PrpsRow1.HLS.Period": 1.5474060773849487`` and generate
    ``"P3.DOV.PrpsRow1.HLS.Period": [1.5474060773849487, "مدت زمان اقامت خوب است"]``


    See Also:

        - XAI module: :mod:`vizard.xai`
        - SHAP module: :mod:`vizard.xai.shap`
        - A mapping of features to basic text: ``vizard.data.constant.FEATURE_NAME_TO_TEXT_MAP``
        - XAI value thresholding method: :meth:`vizard.xai.core.xai_threshold_to_text`

    Args:
        xai_feature_values (Dict[str, float]): XAI values for features
        feature_to_keyword_mapping (Dict[str, str]): Mapping from feature to keyword

    Returns:
        Dict[str, float, str]: XAI values for features with text
    """
    xai_txt_top_k: Dict[str, Tuple[float, str]] = {}
    for _feature_name, _feature_xai_value in xai_feature_values.items():
        xai_txt_top_k[_feature_name] = (
            _feature_xai_value,
            (
                f"{feature_to_keyword_mapping[_feature_name]}"
                f" {xai_threshold_to_text(xai_value=_feature_xai_value, threshold=0.)}"
            ),
        )

    return xai_txt_top_k


def xai_category_texter(
    xai_feature_values: Dict[str, float], feature_to_keyword_mapping: Dict[str, str]
):

    name_map = {
        "P1.PD.DOBYear.Period": "sex",
        "p1.SecA.ParAccomp.Count": "parent_accompany",
        "P1.MS.SecA.DateOfMarr.Period": "marriage_period",
        "p1.SecA.Sps.SpsAccomp.Count": "spouse_accompany",
        "p1.SecB.Chd.X.ChdRel.ChdCount": "child_count",
        "p1.SecA.App.ChdMStatus": "applicant_marital_status",
        "p1.SecB.Chd.X.ChdAccomp.Count": "child_accompany",
        "p1.SecC.Chd.X.ChdAccomp.Count": "sibling_accompany",
        "P3.Occ.OccRow1.Period": "occupation_period",
    }  # TODO: we are already doing this when we get data from api we just need to reverse the process 
    response_explain = {key: [] for key in FEATURE_CATEGORY_TO_FEATURE_NAME_MAP.keys()}

    for _feature_name, _feature_xai_value in xai_feature_values.items():
        for (
            _feature_category,
            _feature_name_list,
        ) in FEATURE_CATEGORY_TO_FEATURE_NAME_MAP.items():
            if _feature_name in _feature_name_list:
                response_explain[_feature_category].append(
                    {
                        "name": (
                            name_map[_feature_name]
                            if _feature_name in name_map
                            else _feature_name
                        ),
                        "value": _feature_xai_value,
                        "txt": f"{feature_to_keyword_mapping[_feature_name]} {xai_threshold_to_text(xai_value=_feature_xai_value, threshold=0.)}",
                    }
                )
    return response_explain
