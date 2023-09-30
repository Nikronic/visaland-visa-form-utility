# ours
from vizard.data import functional

# helpers
from pathlib import Path
from enum import Enum, auto
from typing import List, Dict, Union


# data used for parsing other fields gathered manually including 
#   invitation letter, travel history, financial status, etc.

# Shared utility functions
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
    
    