"""
Contains implementation of functions that could be used for processing data everywhere and
    are not necessarily bounded to a class.

"""
import re
import csv


def dict_summarizer(d: dict, cutoff_term: str, KEY_ABBREVIATION_DICT: dict = None,
 VALUE_ABBREVIATION_DICT: dict = None) -> dict:
    """
    Takes a flattened dictionary and modifies it's keys in a way to shorten them by throwing away
        some part and using a abbreviation dictionary for both keys and values.
    
    args:
        d: The dictionary to be shortened
        cutoff_term: The string that used to find in keys and remove anything behind it
        KEY_ABBREVIATION_DICT: A dictionary containing abbreviation mapping for keys
        VALUE_ABBREVIATION_DICT: A dictionary containing abbreviation mapping for values 
    """

    new_keys = {}
    new_values = {}
    for k,v in d.items():
        if KEY_ABBREVIATION_DICT is not None:
            new_k = k
            if cutoff_term in k:  # FIXME: cutoff part should be outside of abbreviation
                new_k = k[k.index(cutoff_term)+len(cutoff_term)+1:]
            
            ### add any filtering over keys here
            # abbreviation
            for word, abbr in KEY_ABBREVIATION_DICT.items():
                new_k = re.sub(word, abbr, new_k)
            new_keys[k] = new_k
        
        if VALUE_ABBREVIATION_DICT is not None:
            # values can be `None`
            if v is not None:
                new_v = v
                if cutoff_term in v:  # FIXME: cutoff part should be outside of abbreviation
                    new_v = v[v.index(cutoff_term)+len(cutoff_term)+1:]
                
                ### add any filtering over values here
                # abbreviation
                for word, abbr in VALUE_ABBREVIATION_DICT.items():
                    new_v = re.sub(word, abbr, new_v)
                new_values[v] = new_v
            else:
                new_values[v] = v

    # return a new dictionary with updated values
    if KEY_ABBREVIATION_DICT is None:
        new_keys = dict((key, key) for (key, _) in d.items())
    if VALUE_ABBREVIATION_DICT is None:
        new_values = dict((value, value) for (_, value) in d.items())
    return dict((new_keys[key], new_values[value]) for (key, value) in d.items())

def dict_to_csv(d: dict, path: str) -> None:
    """
    Takes a flattened dictionary and writes it to a CSV file.
    
    args:
        d: A dictionary
        path: String to where file should be saved
    """
    
    with open(path, 'w') as f:
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)
