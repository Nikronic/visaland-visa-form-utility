__all__ = [
    'SFSlicing', 'ComposeSFSlicing', 'SinglePerson',
]

# core
import pandas as pd
import re
# snorkel
from snorkel.slicing import slicing_function
from snorkel.slicing import SlicingFunction
# helpers
from typing import Optional, Any, List, Callable, Sequence


class SFSlicing:
    """Adds slicing capabilities unique to a country for ``snorkel.SlicingFunction``

    Notes:
        User must create new class that subclasses this and write domain/dataset
        specific methods for his/her case. For instance, if you need to add
        slicing for a class with "single person" value, then extend this class and
        override :func:`slice` method of it. e.g. :class:`SinglePerson`.

        At the moment, the goal is to use this base to include all core slicing
        methods that could be used anywhere.
        And make sure that any class that subclass this, should integrate 
        ``snorkel.SlicingFunction`` to be usable in ``snorkel`` pipeline.
        For instance, see the following for instance of implementation and usage:

            * :class:`SinglePerson`
            * and so on.

    """

    def __init__(self) -> None:
        self.COLUMN = ''

    def slice(self, s: pd.Series, column: str = None) -> bool:
        """Slices a Pandas Series based on a heuristic

        Args:
            s (pd.Series): A series of our dataframe to be
                conditioned for slicing

        Returns:
            bool: True if condition is met, False otherwise
        """
        raise NotImplementedError

    def make_sf(self, func: Callable,
                class_name: Optional[str] = None,
                **kwargs) -> SlicingFunction:
        """Make any function an instance of ``snorkel.SlicingFunction``

        Note:
            Currently only ``func`` s that work on ``pd.Series`` that work on
            single ``column`` are supported as the API is designed this way.
            But, it could be easily modified to support any sort of function.

        Args:
            func (Callable): A callable that
            column (str): column name of a ``pd.Series`` that is going to be read 
                as the condition of determining the belonging to a slice.
                Must be provided if ``func`` is not setting it internally. E.g. for
                :class:`SinglePerson` you don't need to 
                set ``column`` since it is being handled internally.
            class_name (str, optional): The name of the class if ``func`` is a method of it.
                It is used for better naming given class name alongside ``func`` name.
                Defaults to None.
            kwargs: keyword arguments to ``func``

        Returns:
            SlicingFunction:
            A callable object that is now compatible with
            ``snorkel`` slicing pipeline, e.g. ``Policy`` and ``LFApplier``
        """

        column = kwargs.pop('column')

        if class_name is None:
            return SlicingFunction(
                name=f'{func.__name__}_{column}',
                f=func, resources=dict(column=column, **kwargs),
            )
        else:
            return SlicingFunction(
                name=f'{class_name}_{column}',
                f=func, resources=dict(column=column, **kwargs),
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
            raise ValueError(f'Row must be between {lb} and {ub}, got {row}')

    def check_valid_section(self, section: str, sections: list) -> None:
        """Check if the section is valid. Practically, this is a search in list

        Args:
            section (str): which section of the column
            sections (list): list of valid sections
        Raises:
            ValueError: if ``section`` is not inside the ``sections`` list
        """
        if section not in sections:
            raise ValueError(f'Section must be in {sections}, got {section}')


class ComposeSFSlicing(SFSlicing):
    """Composes a list of :class:`SFSlicing` instances 

    Examples:
        >>> sf_compose = [
        >>>     slicing.SinglePerson(),
        >>> ]
        >>> sfs = slicing.ComposeSFSlicing(slicers=sf_compose)()
        >>> sf_applier = PandasSFApplier(sfs, random_policy)

    """

    def __init__(self, slicers: Sequence[SFSlicing]) -> None:
        super().__init__()

        for slicer in slicers:
            if not issubclass(slicer.__class__, SFSlicing):
                raise TypeError(f'Keys must be instance of {SFSlicing}.')

        self.slicers = slicers

    def __call__(self, *args: Any, **kwds: Any) -> List[SlicingFunction]:
        """Takes a list of :class:`SFSlicing` and converts to ``snorkel.SlicingFunction``

        Returns:
            list[SlicingFunction]:
            A list of objects that instantiate ``snorkel.SlicingFunction``
        """
        slicers_sf: List[SlicingFunction] = []
        for slicer in self.slicers:
            slicer = self.make_sf(func=slicer.slice, column=slicer.COLUMN,
                                  class_name=slicer.__class__.__name__)
            slicers_sf.append(slicer)
        return slicers_sf


class SinglePerson(SFSlicing):
    """Single and unmarried slice

    This is being done by using ``'p1.SecA.App.ChdMStatus'`` which contains
    marital status of the person, i.e. if "single" then ``==7``.
    Also, to further verify this, we check the marriage period by
    verifying that ``'P2.MS.SecA.Period'`` is also zero.

    Returns True if the person is single and unmarried, otherwise False
    """

    def __init__(self) -> None:
        super().__init__()

        self.COLUMN = 'p1.SecA.App.ChdMStatus'
        self.HELPER_COLUMN = 'P2.MS.SecA.Period'

    def slice(self, s: pd.Series, column: str = None) -> bool:
        """Slices a Pandas Series based on a heuristic

        Args:
            s (pd.Series): A series of our dataframe to be
                conditioned for slicing

        Returns:
            bool: True if condition is met, False otherwise
        """

        COLUMN = self.COLUMN
        HELPER_COLUMN = self.HELPER_COLUMN

        condition = (s[COLUMN] == 7) & (
            s[HELPER_COLUMN] == 0)
        return True if condition else False
