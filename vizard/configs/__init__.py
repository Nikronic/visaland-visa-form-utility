__all__ = [
    'CANADA_COUNTRY_CODE_TO_NAME'
]

# ref: https://stackoverflow.com/a/50797410/18971263
import pathlib


# path to all config/db files
parent_dir = pathlib.Path(__file__).parent
DATA_DIR = parent_dir / 'data'
# canada visa form country code to name
#   these forms do not use any standard format and has been added manually
#   originally by Canada officials. Hence, this data has been extracted from
#   their forms.
CANADA_COUNTRY_CODE_TO_NAME = DATA_DIR / 'canada-country-code-to-name.csv'

# Iran city to province name
#   The list of cities and their corresponding province have been extracted
#   from https://en.wikipedia.org/wiki/List_of_cities_in_Iran_by_province
IRAN_PROVINCE_TO_CITY = DATA_DIR / 'iran-province-to-city.csv'
