import re
import pandas as pd
from snorkel.slicing import slicing_function
from snorkel.slicing import slice_dataframe


@slicing_function()
def single_person(x: pd.Series) -> bool:
    """single and unmarried slice

    Args:
        x (pd.Series): input Pandas Series 

    Returns:
        bool: Whether or not `x` satisfies the condition
    """

    #
    condition = (x['P2.MS.SecA.Period'] == 0) & (
        x['p1.SecA.App.ChdMStatus'] == 7)
    return True if condition else False
