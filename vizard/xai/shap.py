# core
import numpy as np
import shap
from flaml import AutoML
# ours
from vizard.xai.core import get_top_k
from vizard.data.constant import (
    FeatureCategories,
    FEATURE_CATEGORY_TO_FEATURE_NAME_MAP
)
# helpers
from typing import Dict, List, Optional
import logging


# configure logging
logger = logging.getLogger(__name__)


class FlamlTreeExplainer:
    """An interface between ``shap`` and ``flaml`` for explaining tree-based models via **SHAP** values
    """
    def __init__(
        self,
        flaml_model: AutoML,
        feature_names: List[str],
        data: Optional[np.ndarray] = None
    ) -> None:
        """Initialize :class:`shap.TreeExplainer` for :class:`flaml.AutoML` tree based models

        Args:
            flaml_model (:class:`flaml.AutoML`): Fitted tree-based ``flaml`` model
            feature_names (List[str]): List of feature names that are preprocessed (features
                used directly to train ``flaml_model``.)
            data (Optional[:class:`numpy.ndarray`]): Optionally to provide for other type of output
                that :class:`shap.TreeExplainer` provides. Defaults to None.
        """

        self.flaml_model = flaml_model
        self.feature_names = feature_names

        # main SHAP object used for inferring XAI values
        self.explainer = shap.TreeExplainer(
            model=self.flaml_model.model.estimator,
            data=data,
            feature_names=feature_names,
        )

    @staticmethod
    def __validate_sample_shape(sample: np.ndarray) -> np.ndarray:
        """Makes sure input sample(s) is 2D

        Note:
            It is expected that this method would be only useful when the sample 
            is a single instance in shape of ``(m, )`` where ``m`` is the number
            of features.

        Args:
            sample (:class:`numpy.ndarray`): A :class:`numpy.ndarray`

        Returns:
            :class:`numpy.ndarray`:
                A :class:`numpy.ndarray` in shape of ``(n, m)``
                where ``n`` are number of samples and ``m`` number of features.
        """

        if sample.ndim == 1:  # shape = (m, )
            sample = sample.reshape(1, -1)

        if sample.ndim != 2:  # sample(s) must be 2D
            raise ValueError(
                f'Given sample is "{sample.ndim}", which must be 2D.')

        return sample
    
    def _get_indices(self, sublist: List[str], superlist: List[str]) -> List[int]:
        """Finds the index of strings of B in A where strings in B have similar initial chars as strings in A

        Note:
            This is used for finding the indices of features that are related to a specific topic.

        Args:
            sublist (List[str]): List of strings as subset of ``superlist``
            superlist (List[str]): List of strings where are shortened versions of strings in 
                ``sublist``.

        Returns:
            List[int]: List of indices of strings of ``sublist`` in ``superlist``
        """

        return [i for item in sublist for i, s_item in enumerate(superlist) if s_item.startswith(item)]


    def overall_score(self, sample: np.ndarray) -> float:
        """Explains the goodness of an sample aggregated over all features

        If the output value is positive, then ``sample`` is from the positive class and vice versa.
        The absolute of the output value determines the intensity of assignment
        to positive (negative) class.

        Note: 
            ``shap_value`` (i.e. ``explainer(X[i,:].reshape(1, -1))`` ),
            is the sum of difference of the output of the model from the expected value of the model. I.e.:

            .. math::
                \\text{shap} = E[f_{\\theta}(x^c_i)] - \sum^C_{c=0} f_{\\theta}(x^c_i)
            

            Where :math:`E[f_{\\theta}(x^c_i)]` is the expected value of model, constant,
            for the entire dataset, :math:`C` the number of features (columns),
            :math:`f_{\\theta}` the fitted model, and :math:`i=0` is a sample in
            the dataset. To use these values for the final user, the expected
            value :math:`E[f_{\\theta}(x^c_i)]` means nothing.
            So, to simplify these values, the equation can be simplified into:

            .. math::
                \\text{shap}_\\text{sim} = (E[f_{\\theta}(x^c_i)] - \sum^C_{c=0} f_{\\theta}(x^c_i)) - E[f_{\\theta}(x^c_i)] = \\text{shap} - E[f_{\\theta}(x^c_i)]
            
        """

        sample = self.__validate_sample_shape(sample)

        # compute shap
        shap_output: shap.Explanation = self.explainer(sample)
        score: np.ndarray = np.sum(shap_output.values) - shap_output.base_values
        return score.item()

    def top_k_score(self, sample: np.ndarray, k: int = 5) -> dict:
        """Returns scores and names of top-k impactful features

        Note:
            By top-k, it means the top-k features that might have negative
            or positive effect. So, the top-k, are top-k absolute values of
            shap values.

        Args:
            sample (:class:`numpy.ndarray`): A :class:`numpy.ndarray`
            k (int, optional): Number of features to include. Defaults to 5.

        Returns:
            dict:
                A dictionary where keys are names of features and values
                are **shap** values for the corresponding features.
        """
        sample = self.__validate_sample_shape(sample)

        # compute shap
        shap_output: shap.Explanation = self.explainer(sample)
        # original shap values
        shap_values: np.ndarray = shap_output.values.flatten()
        # top k shap values with their signs
        top_k_values, top_k_idx = get_top_k(sample=shap_values, k=k)
        # top k feature names for vis purposes
        top_k_feature_names: np.ndarray = np.array(
            self.feature_names)[top_k_idx]
        # gather shap values and feature names into a dictionary
        top_k: Dict[str, float] = {}
        for _value, _feature_name in zip(top_k_values, top_k_feature_names):
            top_k[_feature_name] = _value

        return top_k

    def aggregate_shap_values(
            self,
            sample: np.ndarray,
            feature_category_to_feature_name: Dict[FeatureCategories, List[str]],
            ) -> Dict[FeatureCategories, float]:
        """Aggregates SHAP values into multiple categories

        This method is used to aggregate SHAP values of features, into different
        groups that represent a topic. Then, by let's say summing all the SHAP values 
        of a specific group, we have score for that group. Note that defining features of
        each group is decided by stakeholders and hence,
        a manual list of features for each group should be provided. Argument
        ``feature_category_to_feature_name`` will contain this value.

        Args:
            sample (:class:`numpy.ndarray`): A :class:`numpy.ndarray` representing a sample
            feature_category_to_feature_name (Dict[FeatureCategories, List[str]]): A dictionary
                where keys are category names and values are lists of feature names that
                belong to the category.

        Returns:
            Dict[FeatureCategories, float]:
                A dictionary where keys are category names and values are aggregations 
                of shap values for the corresponding features.
        """
        sample = self.__validate_sample_shape(sample)

        # compute shap
        shap_output: shap.Explanation = self.explainer(sample)
        
        # aggregate shap values via summation
        aggregated_shap_values: Dict[FeatureCategories, float] = {}
        for _category, _features in feature_category_to_feature_name.items():
            # get indices of group features in all features names
            features_idx = self._get_indices(
                sublist=_features,
                superlist=self.feature_names
            )
            # aggregate shap values of group features
            aggregated_shap_values[_category] = np.sum(shap_output.values[:, features_idx])

        return aggregated_shap_values
