__all__ = [
    'FLAMLDTreeViz'
    'trees'
]

# core
from dtreeviz import trees
from xgboost import XGBClassifier
from dtreeviz.models.xgb_decision_tree import ShadowXGBDTree
from lightgbm import LGBMClassifier
from dtreeviz.models.lightgbm_decision_tree import ShadowLightGBMTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from catboost import CatBoostClassifier
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from dtreeviz.models.sklearn_decision_trees import ShadowDecTree
import numpy as np
import flaml
# helpers
from typing import Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class FLAMLDTreeViz:
    """An interface for visualizing flaml_ *tree* based models via dtreeviz_

    Notes:
        FLAML has a wrapper around popular libraries such as ``sklearn``,
        ``XGBoost``, and so on. If you are using one of these libraries directly,
        use ``dtreeviz`` directly instead of using this class or any base class of it.

    Examples:
        See official examples_ from ``dtreeviz`` package.

    .. _flaml: https://github.com/microsoft/FLAML
    .. _dtreeviz: https://github.com/parrt/dtreeviz
    .. _examples: https://colab.research.google.com/github/parrt/dtreeviz/blob/master/notebooks/examples.ipynb
    """

    def __init__(
        self,
        flaml_automl: flaml.AutoML,
        x_data: np.ndarray,
        y_data: Optional[np.ndarray] = None,
        target_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        explanation_type: str = 'plain_english'
    ) -> None:
        """Initialized the interface

        Args:
            flaml_automl (flaml.AutoML): An already fitted :class:`flaml.AutoML` instance
            x_data (:class:`numpy.ndarray`): Input features used for training the model.
                Could be :class:`pandas.Dataframe` or :class:`numpy.ndarray` 2D data.
            y_data (:class:`numpy.ndarray`): Output ground truth labels with same length
                as ``x_data``. In case of binary classification, it should
                be flattened (``shape=(n, )``)
            target_name (str, optional): Name of labels
            feature_names (List[str]): A list of names corresponding to the features of ``x_data``
            class_names (List[str], optional): Name of classes corresponding to unique
                labels in ``y_data``
        """
        self.flaml_automl: flaml.AutoML = flaml_automl
        self.x_data: np.ndarray = x_data
        self.y_data: Optional[np.ndarray] = y_data
        self.__target_name: Optional[str] = target_name
        self.__feature_names: Optional[List[str]] = feature_names
        self.__class_names: Optional[List[str]] = class_names
        self.explanation_type: str = explanation_type

        self._find_estimator()

    @property
    def feature_names(self) -> List[str]:
        if self.__feature_names is None:
            self.__feature_names = self.flaml_automl.feature_names_in_
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, _feature_names: List[str]):
        self.__feature_names = _feature_names

    @property
    def target_name(self) -> str:
        if self.__target_name is None:
            self.__target_name = 'target'
        return self.__target_name

    @property
    def class_names(self) -> List[str]:
        if self.__class_names is None:
            unique_classes = np.unique(np.array(self.y_data))
            _class_names = [f'c{i}' for i in range(len(unique_classes))]
            self.__class_names = _class_names
        return self.__class_names

    def _find_estimator(self) -> Any:
        """Finds underlying fitted estimator from :attr:`flaml_automl` as the tree for ``dtreeviz``

        Raises:
            NotImplementedError: If ``flaml_automl`` underlying ``model.estimator`` is not
                implemented yet

        Returns:
            Any:
            The underlying estimator itself or a component of it that is going to be used
            as the tree model for different visualization methods.
        """
        # TODO: run flaml for each estimator separately to find underlying estimator type

        # get best flaml model
        estimator: Any = self.flaml_automl.model.estimator
        # find type of estimator
        if isinstance(estimator, XGBClassifier):
            estimator = estimator  # TODO: e.g. of that!
            # check this
            estimator = estimator.get_booster()
        elif isinstance(estimator, LGBMClassifier):
            estimator = estimator.booster_
        elif isinstance(estimator, RandomForestClassifier):
            estimator = estimator  # TODO: e.g. of that!
        elif isinstance(estimator, CatBoostClassifier):
            raise NotImplementedError('Weirdge, not implemented')
            estimator = estimator  # TODO: does not work at all
        elif isinstance(estimator, DecisionTreeClassifier):
            estimator = estimator
        elif isinstance(estimator, ExtraTreeClassifier):
            estimator = estimator
        else:
            raise NotImplementedError(
                f'estimator "{estimator.__class__.__name__}" not found')
        self.estimator = estimator
        return estimator

    @staticmethod
    def __plot(
        plot: Optional[Path] = None,
        name: Optional[str] = None
    ) -> None:
        """Shows the plot or saves to the disk

        Args:
            plot (Optional[Path], optional): Whether or not to plot the result.
                If ``plot`` is a :class:`pathlib.Path`, then plot will be saved
                in that path, otherwise ``plt.show`` would be called.. Defaults to None.
            name (Optional[str], optional): Name of the plot if ``plot`` is a 
                :class:`pathlib.Path`. Defaults to None.

        Raises:
            ValueError: If ``plot`` is provided but ``name`` is None
        """
        if isinstance(plot, Path):
            if name is None:
                raise ValueError(
                    '`name` has to be provided when `plot` is not none'
                )
            plt.savefig(plot / name)
        else:
            plt.show()

    def viz_leaf_samples(
        self,
        tree_index: int,
        plot: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 5),
        *args, **kwargs
    ) -> None:
        """A wrapper around :func:`dtreeviz.trees.viz_leaf_samples`

        Args:
            tree_index (int): The index of tree in ensemble tree methods
            plot (Path, optional): Whether or not to plot the result.
                If ``plot`` is a :class:`pathlib.Path`, then plot will be saved
                in that path, otherwise ``plt.show`` would be called. Defaults to None.
            figsize (Tuple[int, int], optional): Size of plot figure. 
        """
        self.estimator
        plt.clf()
        trees.viz_leaf_samples(
            tree_model=self.estimator,
            tree_index=tree_index,
            x_data=self.x_data,
            feature_names=self.feature_names,
            figsize=figsize,
            *args,
            **kwargs
        )
        fig_name: str = f'viz_leaf_samples_tree_{tree_index}.svg'
        self.__plot(plot=plot, name=fig_name)

    def dtreeviz(
        self,
        tree_index: int,
        scale: float = 1.0,
        orientation: str = 'TD',
        plot: Optional[Path] = None,
        *args, **kwargs
    ) -> None:
        """A wrapper around :func:`dtreeviz.trees.dtreeviz`

        Args:
            tree_index (int): The index of tree in ensemble tree methods
            scale (float, optional): Scale the figure. Defaults to 1.0.
            orientation (str, optional): Orientation of graph, top-down ``'TD'`` or
                left-right ``'LR'``. Defaults to 'TD'.
            plot (Optional[Path], optional): Whether or not to plot the result.
                If ``plot`` is a :class:`pathlib.Path`, then plot will be saved
                in that path, otherwise ``plt.show`` would be called. Defaults to None.
        """
        plt.clf()
        trees.dtreeviz(
            tree_model=self.estimator,
            tree_index=tree_index,
            x_data=self.x_data,
            y_data=self.y_data,
            feature_names=self.feature_names,
            target_name=self.target_name,
            class_names=self.class_names,
            orientation=orientation,
            scale=scale,
            *args,
            **kwargs
        )
        fig_name: str = f'dtreeviz_tree_{tree_index}.svg'
        self.__plot(plot=plot, name=fig_name)

    def explain_prediction_path(
        self,
        tree_index: int,
        instance_index: int,
        plot: Optional[Path] = None,
        explanation_type: str = 'plain_english',
        *args, **kwargs
    ) -> Optional[str]:
        """A wrapper around :func:`dtreeviz.trees.explain_prediction_path`

        Args:
            tree_index (int): The index of tree in ensemble tree methods
            instance_index (int): A single instance of :attr:`x_data`
            plot (Optional[:class:`pathlib.Path`], optional): Whether or not to plot the result.
                If ``plot`` is a :class:`pathlib.Path`, then plot will be saved
                in that path, otherwise ``plt.show`` would be called. Defaults to None.
            explanation_type (str, optional): Explanation format, ``'plain_english'`` (string)
                or ``'sklearn_default'`` (image). Defaults to 'plain_english'.

        Raises:
            ValueError: If ``'sklearn_image'`` is chosen but ``plot`` is None
            ValueError: If ``explanation_type`` is unknown

        Returns:
            Optional[str]: string if ``explanation_type='plain_english'`` otherwise None
        """

        # uses matplotlib figure
        if explanation_type == 'sklearn_default':
            if plot is None:
                raise ValueError('`plot` cannot be none when'
                                 " `explanation_type` is `'sklearn_default'`")
            plt.clf()
        if explanation_type == 'plain_english':
            if plot is not None:
                logger.warning("`'explanation_type'` is to 'plain_english' but `plot` is provided too"
                               ' hence, it will be ignored and a string will be returned.')
        elif explanation_type not in ['plain_english', 'sklearn_default']:
            raise ValueError(
                f'`explanation_type` "{explanation_type}" is not valid'
            )

        explanation = trees.explain_prediction_path(
            tree_model=self.estimator,
            x=self.x_data[instance_index],
            x_data=self.x_data,
            y_data=self.y_data,
            feature_names=self.feature_names,
            target_name=self.target_name,
            class_names=self.class_names,
            tree_index=tree_index,
            explanation_type=explanation_type,
            *args,
            **kwargs
        )

        # return string as plain english
        if explanation_type == 'plain_english':
            return explanation
        # save sklearn mode plot
        else:
            fig_name: str = f'explain_prediction_path_index_{instance_index}_tree_{tree_index}.svg'
            self.__plot(plot=plot, name=fig_name)
            return None
