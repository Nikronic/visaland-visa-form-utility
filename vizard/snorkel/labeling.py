__all__ = [
    "labeling_function",
    "ComposeLFLabeling",
    "LFLabeling",
    "WeakAccept",
    "WeakReject",
    "NoIdea",
    "ABSTAIN",
    "REJ",
    "ACC",
]

# core
import pandas as pd

# snorkel
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction

# helper
from typing import Optional, Any, List, Callable, Sequence
import logging


# configure logging
logger = logging.getLogger(__name__)

# TODO: move this to `vizard.snorkel.constant.py`
# define the label mappings
ABSTAIN = -1
REJ = 0  # TODO: cant be 2 so it matches our dataframe already?
ACC = 1


class LFLabeling:
    """Adds labeling capabilities unique to a country for ``snorkel.LabelingFunction``

    Notes:
        User must create new class that subclasses this and write domain/dataset
        specific methods for his/her case. For instance, if you need to add
        labeling to a class with "age" value, then extend this class and
        override :func:`label` method of it. e.g. :class:`WeakAccept`.

        At the moment, the goal is to use this base to include all core labeling
        methods that could be used anywhere.
        And make sure that any class that subclass this, should integrate
        ``snorkel.LabelingFunction`` to be usable in ``snorkel`` pipeline.
        For instance, see the following for instance of implementation and usage:

            * :class:`WeakAccept`
            * :class:`WeakReject`
            * :class:`NoIdea`
            * and so on.

    """

    def __init__(self) -> None:
        self.COLUMN = ""

    def label(self, s: pd.Series, column: str = None) -> int:
        """Labels a Pandas Series based on a heuristic

        Args:
            s (:class:`pandas.Series`): An unlabeled series of our dataframe to be labeled

        Returns:
            int: Labeling result
        """
        raise NotImplementedError

    def make_lf(
        self, func: Callable, class_name: Optional[str] = None, **kwargs
    ) -> LabelingFunction:
        """Make any function an instance of ``snorkel.LabelingFunction``

        Note:
            Currently only ``func`` s that work on :class:`pandas.Series` that work on
            single ``column`` are supported as the API is designed this way.
            But, it could be easily modified to support any sort of function.

        Args:
            func (Callable): A callable that
            column (str): column name of a :class:`pandas.Series` that is going to be read
                as the condition of determining the label.
                Must be provided if ``func`` is not setting it internally. E.g. for
                :class:`WeakAccept` you don't need to
                set ``column`` since it is being handled internally.
            class_name (str, optional): The name of the class if ``func`` is a method of it.
                It is used for better naming given class name alongside ``func`` name.
                Defaults to None.
            kwargs: keyword arguments to ``func``

        Returns:
            LabelingFunction:
            A callable object that is now compatible with
            ``snorkel`` labeling pipeline, e.g. ``Policy`` and ``LFApplier``
        """

        column = kwargs.pop("column")

        if class_name is None:
            return LabelingFunction(
                name=f"{func.__name__}_{column}",
                f=func,
                resources=dict(column=column, **kwargs),
            )
        else:
            return LabelingFunction(
                name=f"{class_name}_{column}",
                f=func,
                resources=dict(column=column, **kwargs),
            )

    def check_valid_row(self, row: int, lb: int, ub: int) -> None:
        """Check if the row is valid

        Args:
            row (int): which row to use for the categorical noise. ``row`` here
                means the ``row`` th column of the dataframe with the same name
                of the column to be noisy.
            lb (int): lower bound of the row
            ub (int): upper bound of the row

        Raises:
            ValueError: if ``row`` is not inside the bounds
        """
        if row < lb or row > ub:
            raise ValueError(f"Row must be between {lb} and {ub}, got {row}")

    def check_valid_section(self, section: str, sections: list) -> None:
        """Check if the section is valid. Practically, this is a search in list

        Args:
            section (str): which section of the column
            sections (list): list of valid sections
        Raises:
            ValueError: if ``section`` is not inside the ``sections`` list
        """
        if section not in sections:
            raise ValueError(f"Section must be in {sections}, got {section}")

    def __repr__(self) -> str:
        msg = (
            f'LabelingFunction "{self.__class__.__name__}" is being used'
            f' on column "{self.COLUMN}"'
        )
        return msg


class ComposeLFLabeling(LFLabeling):
    """Composes a list of :class:`LFLabeling` instances

    Examples:
        >>> lf_compose = [
        >>>     labeling.WeakAccept(dataframe=data)
        >>> ]
        >>> lfs = labeling.ComposeLFLabeling(labelers=lf_compose)()
        >>> lf_applier = PandasLFApplier(lfs, random_policy)

    """

    def __init__(self, labelers: Sequence[LFLabeling]) -> None:
        super().__init__()

        for labeler in labelers:
            if not issubclass(labeler.__class__, LFLabeling):
                raise TypeError(f"Keys must be instance of {LFLabeling}.")

        self.labelers = labelers

        # set logger
        self.logger = logging.getLogger(logger.name + ".ComposeLFLabeling")

        # log the labelers
        self.logger.info(f"Following labeling functions are being used:")
        for labeler in self.labelers:
            self.logger.info(f"* {labeler}")

    def __call__(self, *args: Any, **kwds: Any) -> List[LabelingFunction]:
        """Takes a list of :class:`LFLabeling` and converts to ``snorkel.LabelingFunction``

        Returns:
            list[LabelingFunction]:
            A list of objects that instantiate ``snorkel.LabelingFunction``
        """
        labelers_lf: List[LabelingFunction] = []
        for labeler in self.labelers:
            labeler = self.make_lf(
                func=labeler.label,
                column=labeler.COLUMN,
                class_name=labeler.__class__.__name__,
            )
            labelers_lf.append(labeler)
        return labelers_lf


class WeakAccept(LFLabeling):
    """LabelingFunction to convert *weak accept* to *accept*

    In our dataset, following labels are used:

        * ``'acc'``: accept
        * ``'rej'``: reject
        * ``'w-acc'``: weak accept
        * ``'w-rej'``: weak reject
        * ``'no-idea'``: no idea

    Returns ``ACC`` if the label is ``'w-acc'``, otherwise ``ABSTAIN``
    """

    def __init__(self) -> None:
        super().__init__()

        self.COLUMN = "VisaResult"

    def label(self, s: pd.Series, column: str = None) -> int:
        """Labels a Pandas Series based on a heuristic

        Args:
            s (:class:`pandas.Series`): An unlabeled series of our dataframe to be labeled

        Returns:
            int: Labeling result
        """

        COLUMN = self.COLUMN
        if s[COLUMN] == "w-acc":  # 3 == weak acc
            return ACC
        else:
            return ABSTAIN


class WeakReject(LFLabeling):
    """LabelingFunction to convert *weak reject* to *reject*

    In our dataset, following labels are used:

        * ``'acc'``: accept
        * ``'rej'``: reject
        * ``'w-acc'``: weak accept
        * ``'w-rej'``: weak reject
        * ``'no-idea'``: no idea

    Returns Returns ``REJ`` if the label is ``'w-rej'``, otherwise ``ABSTAIN``.
    """

    def __init__(self) -> None:
        super().__init__()

        self.COLUMN = "VisaResult"

    def label(self, s: pd.Series, column: str = None) -> int:
        """Labels a Pandas Series based on a heuristic

        Args:
            s (:class:`pandas.Series`): An unlabeled series of our dataframe to be labeled

        Returns:
            int: Labeling result
        """

        COLUMN = self.COLUMN
        if s[COLUMN] == "w-rej":  # 4 == weak rej
            return REJ
        else:
            return ABSTAIN


class NoIdea(LFLabeling):
    """LabelingFunction to convert *no idea* to *reject*

    Note:
        Ms. S's suggestion was that if she can't remember,
        then it's probably a `REJ` (=rejected) case

    In our dataset, following labels are used:

        * ``'acc'``: accept
        * ``'rej'``: reject
        * ``'w-acc'``: weak accept
        * ``'w-rej'``: weak reject
        * ``'no-idea'``: no idea

    Returns ``REJ`` if the label is ``'no idea'``, otherwise ``ABSTAIN``.
    """

    def __init__(self) -> None:
        super().__init__()

        self.COLUMN = "VisaResult"

    def label(self, s: pd.Series, column: str = None) -> int:
        """Labels a Pandas Series based on a heuristic

        Args:
            s (:class:`pandas.Series`): An unlabeled series of our dataframe to be labeled

        Returns:
            int: Labeling result
        """

        COLUMN = self.COLUMN
        if s[COLUMN] == "no idea":  # 5 == no idea
            return REJ
        else:
            return ABSTAIN
