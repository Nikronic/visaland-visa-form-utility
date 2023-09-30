# ours
from vizard.data import functional

# helpers
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Union


# data used for parsing other fields gathered manually including 
#   invitation letter, travel history, financial status, etc.

# Shared utility functions
class DatabaseConstants:
    """Keys used in 3rd party "other" data provider.

    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history

    Note:
        This is just to simplify accessing key and values in the JSON of 
        response of 3rd party data provider.
    """
    LITERAL_DATA: str = 'data'
    LITERAL_FIELDS: str = 'fields'
    LITERAL_VALUE: str = 'value'
    LITERAL_UNRAVEL: int = 0


class TravelHistory(DatabaseConstants):
    """'TravelHistory' subsection of :class:`Extensive` for "other" data

    Class variable description:

        * ``key``: key inside the 3rd-party JSON
        * ``COUNTRY``: country name of traveling destination
        * ``START_DATE``: starting date of travel
        * ``END_DATE``: ending date of travel

    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history
    
    """
    key: int = 19
    COUNTRY: int = 0
    START_DATE: int = 1
    END_DATE: int = 2


class Inviter(DatabaseConstants):
    """'Inviter' subsection of :class:`Extensive` for "other" data

    Class variable description are keys in 3rd-party provided JSON:

        * ``key``: key of the class itself (i.e., :class:`Inviter`)
        * ``NAME``: name of the inviting person
        * ``RELATION``: the relation of inviting person to the user
        * ``ADDRESS``: address of the inviting person
    
    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history
    """
    key: int = 12
    NAME: int = 0
    RELATION: int = 1
    ADDRESS: int = 2


class Financial(DatabaseConstants):
    """'Financial' subsection of :class:`Documents` for "other" data

    Class variable description are keys in 3rd-party provided JSON:

        * ``key``: key inside the 3rd-party JSON
        * ``BANK_BALANCE``: bank balance of user
        * ``REAL_ESTATE_COUNT``: number of owned real estate by user

    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history
    """
    key: int = 29
    BANK_BALANCE: int = 8
    REAL_ESTATE_COUNT: int = 9


class Documents(DatabaseConstants):
    """'Documents' subsection of :class:`InformationCategories` for "other" data

    Class variable description are keys in 3rd-party provided JSON:

        * ``key``: key inside the 3rd-party JSON
        * ``FINANCIAL`` (See :class:`Financial`): Contains financial info of the user
            including bank balance and real estate count.

    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history
    """
    key: int = 4
    FINANCIAL: Financial = Financial()


class Extensive(DatabaseConstants):
    """'Extensive' subsection of :class:`InformationCategories` for "other" data

    Class variable description are keys in 3rd-party provided JSON:

        * ``key``: key inside the 3rd-party JSON
        * ``HAS_INVITATION``: Used to check if user has invitation
        * ``INVITER`` (See :class:`Inviter`): Contains info of the inviter 
            of the user
        * ``TRAVEL_HISTORY`` (See :class:`TravelHistory`): Contains the country, 
            start date, and end date of travel histories

    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history
    """
    key: int = 1
    HAS_INVITATION: int = 11
    INVITER: Inviter = Inviter()
    TRAVEL_HISTORY: TravelHistory = TravelHistory()


class InformationCategories(DatabaseConstants):
    """Categorize of information provided by 3rd-party service for "other" data

    Class variables are keys in 3rd-party provided JSON.
    
    Note:
        "other" refers to:

            * invitation letter and relation
            * financial status
            * travel history
    """
    BASIC: int = 0
    EXTENSIVE: Extensive = Extensive()
    FAMILY: int = 2
    ACCOUNT: int = 3
    DOCUMENTS: Documents = Documents()


class VisalandImportUser:
    """An interface for connecting to Visaland Management service

    This service provides the following data which we call "other":

        * invitation letter and relation
        * financial status
        * travel history
    """
    def __init__(self, path: Union[str, Path]) -> None:
        """Initiates the class by reading the json response of the "other" data provider

        Args:
            path (Union[str, Path]): Path to the JSON file
        """
        self.path = path.as_posix() if isinstance(path, Path) else path
        self.data = self._json_to_dict(self.path)
    
    def _json_to_dict(self, path: Union[str, Path] = None) -> Dict:
        """A wrapper around :func:`vizard.data.functional.json_to_dict`

        Args:
            path (Union[str, Path], optional): Path to JSON file.
                Defaults to None.

        Returns:
            Dict: Dictionary associated with the JSON file
        """
        return functional.json_to_dict(path=path)
    
    