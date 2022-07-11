__all__ = [
    'single_person'
]

# core
import pandas as pd
import re
# snorkel
from snorkel.slicing import slicing_function
from snorkel.slicing import slice_dataframe


@slicing_function()
def single_person(x: pd.Series) -> bool:
    """single and unmarried slice

    This is being done by using ``''p1.SecA.App.ChdMStatus''`` which contains
        marital status of the person, i.e. if "single" then ``==7``.
    Also, to further verify this, we check the marriage period by
        verifying that ``''P2.MS.SecA.Period''`` is zero.

    Args:
        x (pd.Series): input Pandas Series 

    Returns:
        bool: True if `x` is a single person, False otherwise
    """

    #
    condition = (x['P2.MS.SecA.Period'] == 0) & (
        x['p1.SecA.App.ChdMStatus'] == 7)
    return True if condition else False
