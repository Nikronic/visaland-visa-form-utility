from .core import JsonConfigHandler

# ref: https://stackoverflow.com/a/50797410/18971263
import pathlib


# path to all config/db files
parent_dir = pathlib.Path(__file__).parent
DATA_DIR = parent_dir / 'data'


CANADA_COUNTRY_CODE_TO_NAME = DATA_DIR / 'canada-country-code-to-name.csv'
"""Canada visa form country code to name

Note:
    These forms do not use any standard format and has been added manually
    originally by Canada officials. Hence, this data has been extracted from
    their forms.
"""

IRAN_PROVINCE_TO_CITY = DATA_DIR / 'iran-province-to-city.csv'
"""Iran city to province name

Note:
    The list of cities and their corresponding province have been extracted
    from https://en.wikipedia.org/wiki/List_of_cities_in_Iran_by_province
"""

COUNTRIES_NAMES_FA2ENG = DATA_DIR / 'countries_fa2eng.csv'
"""The name of the countries from Persian (Fa) to English (En)

Note:
    The list of countries have been obtained from Wikipedia at 
    https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area
    and modified slightly (ignored territories of mainland countries)
"""
