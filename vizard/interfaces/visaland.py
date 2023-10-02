# ours: config
from vizard.configs import (
    CANADA_COUNTRY_CODE_TO_NAME,
    TOURISM_WORLD_REGIONS,
    COUNTRIES_NAMES_FA2ENG
)
# ours: data
from vizard.data import functional
from vizard.data.constant import (
    CanadaContactRelation,
    VisaWorldRegions
)

# helpers
from enum import Enum
from pathlib import Path
from typing import List, Dict, Union, Optional


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
    LITERAL_STATE: str = 'state'
    LITERAL_UNRAVEL: int = 0


class DashboardProcedureState(Enum):
    """The step of the procedure the user is currently in on 3rd party "other" data provider/

    Class variable descriptions:
        
        * ``CONFIRMATION_CONTRACT``: TBD
        * ``CONFIRMATION_ACCOUNTING``: TBD
        * ``CONFIRMATION_BY_EXECUTIVE_EXPERT``: TBD
        * ``CONFIRMATION_CHECKOUT_BY_COUNTER``: TBD
        * ``CONFIRMATION_CHECKOUT_BY_ADMIN``: TBD
        * ``SENDING_FINGERPRINT_DOCUMENT``: TBD
        * ``VISA_ACCEPT``: User's visa has been accepted
        * ``VISA_REJECT``: User's visa has been rejected
    
    Note:
        This is not similar to other documents that have had official structure which then
        we built interfaces on top of that. This only exist in our system and should be
        customized for your own case.
    """

    CONFIRMATION_CONTRACT: int = 1
    CONFIRMATION_ACCOUNTING: int = 2
    CONFIRMATION_BY_EXECUTIVE_EXPERT: int = 3
    CONFIRMATION_CHECKOUT_BY_COUNTER: int = 4
    CONFIRMATION_CHECKOUT_BY_ADMIN: int = 5
    SENDING_FINGERPRINT_DOCUMENT: int = 6
    VISA_ACCEPT: int = 7
    VISA_REJECT: int = 8


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

        self.has_invitation: Optional[bool] + None
    
    def _json_to_dict(self, path: Union[str, Path] = None) -> Dict:
        """A wrapper around :func:`vizard.data.functional.json_to_dict`

        Args:
            path (Union[str, Path], optional): Path to JSON file.
                Defaults to None.

        Returns:
            Dict: Dictionary associated with the JSON file
        """
        return functional.json_to_dict(path=path)
    
    def get_real_estate_count(self, raw: bool = False) -> Union[str, int]:
        """Obtains the number of real estates owned by provided data

        If ``raw`` is ``False``, conversion to integer is being done.
        
        Args:
            raw (bool, optional): If ``True``, will provide the raw value
                directly provided by the 3rd-party provider. Defaults to False.

        Returns:
            Union[str, int]: The number of real estate owned
        """
        data = self.data
        real_estate_count_raw: str = \
            data[InformationCategories.LITERAL_DATA] \
                [InformationCategories.DOCUMENTS.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.DOCUMENTS.FINANCIAL.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.LITERAL_UNRAVEL] \
                [InformationCategories.DOCUMENTS.FINANCIAL.REAL_ESTATE_COUNT] \
                [InformationCategories.LITERAL_VALUE]

        if raw:
            return real_estate_count_raw
        real_estate_count: int = int(real_estate_count_raw)
        return real_estate_count
    
    
    @staticmethod
    def _bank_balance_normalizer(balance: float) -> float:
        """``Vizard``'s custom normalization for IRR to CAD

        Note:
            This is no standard method! Use on your own!
        
        Args:
            balance (float): Balance in IRR

        Returns:
            float: Normalized IRR to CAD
        """

        IRR_TO_CAD_RATIO: float = 1e-6
        IRR_TO_CAD_NORMALIZER_FACTOR: float = 10. * IRR_TO_CAD_RATIO
        bank_balance: float = balance * IRR_TO_CAD_NORMALIZER_FACTOR
        return bank_balance

    def get_bank_balance(self, raw: bool = False) -> Union[str, float]:
        """Obtains the bank balance of user by provided data

        If ``raw`` is ``False``, conversion to exchange rate, then a multiplicative
        factor is added as a normalizer. 

        Note:
            If you plan to use any other normalization (including z-score and so on),
            please use ``raw=True``.

        Note:
            You can override internal normalizer by overriding :func:`_bank_balance_normalizer`.
        
        Args:
            raw (bool, optional): If ``True``, will provide the raw value
                directly provided by the 3rd-party provider. Defaults to False.

        Returns:
            Union[str, float]: The (customized normalized) bank balance
        """
        data = self.data
        bank_balance_raw: str = \
            data[InformationCategories.LITERAL_DATA] \
                [InformationCategories.DOCUMENTS.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.DOCUMENTS.FINANCIAL.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.LITERAL_UNRAVEL] \
                [InformationCategories.DOCUMENTS.FINANCIAL.BANK_BALANCE] \
                [InformationCategories.LITERAL_VALUE]
        
        if raw:
            return bank_balance_raw
        bank_balance: float = self._bank_balance_normalizer(float(bank_balance_raw))
        return bank_balance

    def get_invitation_status(self, raw: bool = False) -> Union[str, bool]:
        """Obtains if the user has invitation letter or not by provided data

        If ``raw`` is ``False``, conversion to boolean is being done.

        Args:
            raw (bool, optional): If ``True``, will provide the raw value
                directly provided by the 3rd-party provider. Defaults to False.

        Returns:
            Union[str, bool]: The invitation binary status.
        """
        data = self.data
        has_invitation_raw: str = \
            data[InformationCategories.LITERAL_DATA] \
                [InformationCategories.EXTENSIVE.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.EXTENSIVE.HAS_INVITATION] \
                [InformationCategories.LITERAL_VALUE]
        
        if raw:
            return has_invitation_raw
        has_invitation: bool = True if has_invitation_raw == '1' else False

        # set in class level for `get_invitation_relation`
        self.has_invitation = has_invitation
        return has_invitation

    def get_invitation_relation(self, raw: bool = False) -> str:
        """Obtain the relation of the inviter of the user by provided data

        If ``raw`` is ``False``, the input string of relation is converted
        to :class:`Enum` classes defined in ``vizard``, 
        i.e., :class:`vizard.data.constant.CanadaContactRelation`.

        Args:
            raw (bool, optional): If ``True``, will provide the raw value
                directly provided by the 3rd-party provider. Defaults to False.

        Returns:
            str: 
                The relation, either raw string provided by user or processed
                via :class:`vizard.data.constant.CanadaContactRelation`.
        """

        if not self.has_invitation:
            raise ValueError(
                f'User has no invitation letter. Make sure you already'
                f' have called method `get_invitation_status` first!')
        
        data = self.data
        invitation_relation_raw: str = \
            data[InformationCategories.LITERAL_DATA] \
                [InformationCategories.EXTENSIVE.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.EXTENSIVE.INVITER.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.LITERAL_UNRAVEL] \
                [InformationCategories.EXTENSIVE.INVITER.RELATION] \
                [InformationCategories.LITERAL_VALUE]
        
        if raw:
            return invitation_relation_raw
        
        # TODO: move to outside as a constant
        rel_cat = {
            # order matters, put weaker on top, i.e. put 'law' above 'brother',
            #  so 'brother in law' get handled by 'law' rule than 'bother' rule
            'law': 'f2',
            'nephew': 'f2',
            'niece': 'f2',
            'aunt': 'f2',
            'uncle': 'f2',
            'cousin': 'f2',
            'relative': 'f2',
            'grand': 'f2',
            'parent': 'f1',
            'mother': 'f1',
            'father': 'f1',
            'child': 'f1',
            'daughter': 'f1',
            'brother': 'f1',
            'sister': 'f1',
            'wife': 'f1',
            'husband': 'f1',
            'step': 'f1',
            'son': 'f1',
            'partner': 'f1',
            'fiance': 'f1',
            'fiancee': 'f1',
            'other': 'ukn',
            'friend': 'friend',
            'league': 'work',
            'symposium': 'work',
            'hote': 'hotel',
            'hotel': 'hotel',
        }

        # TODO: make it general (in functional) or class function.
        def fix_rel(
                string: str,
                dic: dict,
                if_nan: str = 'ukn'):
            string = string.lower()
            for k, v in dic.items():
                if k in string:
                    string = string.replace(string, v)
                    return string
            return if_nan

        invitation_relation: CanadaContactRelation = fix_rel(
            string=invitation_relation_raw.lower(),
            dic=rel_cat,
            if_nan='ukn'
        )

        return invitation_relation

    def get_travel_history(
            self,
            raw: bool = False,
            lang: str = 'en'
        ) -> Union[Union[Dict[VisaWorldRegions, int], float], List]:
        """Obtain the countries visited as a travel history of the user by provided data

        If ``raw`` is ``True``, a chronological list (including duplicates)
        containing names of the countries will be returned. 
        
        If ``raw`` is ``False``,
        then we first (create/)use our own world regions based on the hardship of getting
        a visa as defined in :class:`VisaWorldRegions`. Then via some domain expert knowledge
        that assigned (See :var:`vizard.interface.data.TOURISM_WORLD_REGIONS`) countries
        to each of the regions defined in :class:`VisaWorldRegions`. Finally,
        we count how many times the user have visited countries of each region.
        In the end, we return a dictionary of regions of the count of visitation; along
        side the total score i.e., weighted sum of regions (weights are defined in
        :class:`VisaWorldRegions`)

        Args:
            raw (bool, optional): If ``True``, will provide the raw value
                directly provided by the 3rd-party provider. Defaults to False.
            lang (str, optional): The language used for countries names. The target
                language will be English ``lang='en'``. Defaults to ``'en'``.

        Returns:
            List: A list of countries names chronologically
        """
        data = self.data

        if lang not in ['en', 'fa']:
            raise ValueError(f'"lang={lang}" is not recognized.'
                             f'Please use `"en"` for `"fa"`.')

        # visa world regions
        tourism_world_regions: Dict[str, List[str]] = functional.json_to_dict(
            TOURISM_WORLD_REGIONS
        )

        # converting countries name from Fa to English
        countries_names_fa2eng: Dict[str, str] = functional.config_csv_to_dict(
            COUNTRIES_NAMES_FA2ENG
        )
        
        # we don't care for start date and end date of travel
        #   which are recorded in `TravelHistory.START_DATE` and `TravelHistory.END_DATE`
        travel_history_container: List = \
            data[InformationCategories.LITERAL_DATA] \
                [InformationCategories.EXTENSIVE.key] \
                [InformationCategories.LITERAL_FIELDS] \
                [InformationCategories.EXTENSIVE.TRAVEL_HISTORY.key] \
                [InformationCategories.LITERAL_FIELDS]
        # ordered list of countries names
        travel_history_raw: List[str] = []
        for travel in travel_history_container:
            travel_history_raw.append(
                travel[InformationCategories.EXTENSIVE.TRAVEL_HISTORY.COUNTRY] \
                    [InformationCategories.LITERAL_VALUE]
            )
        # verify country name is correct
        #   note that this step could be entirely ignored since we only care about those regions
        #   but since people have many issues in "typing" the correct name, it is better to double check
        country_names: List[str] = list(
            functional.config_csv_to_dict(CANADA_COUNTRY_CODE_TO_NAME).values()
        )

        # list for translated names of countries
        travel_history_raw_trans: List[str] = []

        # extract country names
        for travel in travel_history_container:
            # get country name in source lang
            travel_country: str = \
                travel[InformationCategories.EXTENSIVE.TRAVEL_HISTORY.COUNTRY] \
                    [InformationCategories.LITERAL_VALUE]
            
            # translate from Farsi to English
            if lang == 'fa':
                # change country name to English (from Persian)
                travel_country = countries_names_fa2eng.get(travel_country, travel_country)
                # check if country is valid (typo and stuff). Count as "other" in your scenario
                if travel_country not in country_names:
                    print(f'Country "{travel_country}" is not valid!')
            travel_history_raw_trans.append(travel_country)
        
        travel_history_raw = travel_history_raw_trans
        if raw:
            return travel_history_raw
        
        # count based on region
        travel_history_regions: Dict[VisaWorldRegions, int] = {}
        for travel_country in travel_history_raw:
            # return the region of the country (key of dict of region:country)
            region = next(
                (_region for _region, _countries in tourism_world_regions.items() \
                if travel_country in _countries),
                VisaWorldRegions.LITERAL_OTHER
                )
            
            # count each region
            if region in travel_history_regions:
                travel_history_regions[region] += 1
            else:
                travel_history_regions[region] = 1
        
        # compute travel history score
        travel_history_score: float = 0.
        for region in travel_history_regions:
            travel_history_score += travel_history_regions[region] * VisaWorldRegions[region]

        return (
            travel_history_regions,  # region:count
            travel_history_score     # total score
        )

    def get_visa_result(self, raw: bool = False) -> Union[str, DashboardProcedureState]:
        """Obtain the visa result of the user by provided data

        If ``raw`` is ``False``, the conversion to vizard's internal states (which is in fact
        adopted from the 3rd party provider) will be done. Otherwise, it will be an integer
        defined in the 3rd party provider.

        Note:
            This is the label (dependent variable) in an analytical system.

        Args:
            raw (bool, optional): If ``True``, will provide the raw value
                directly provided by the 3rd-party provider. Defaults to False.

        Returns:
            Union[str, DashboardProcedureState]: The status of the user's visa
        """

        data = self.data

        # getting the visa result
        visa_result_raw: int = \
            data[InformationCategories.LITERAL_DATA] \
                [InformationCategories.LITERAL_STATE]
        
        if raw:
            return visa_result_raw
        
        visa_result: DashboardProcedureState = DashboardProcedureState(visa_result_raw)
        return visa_result
