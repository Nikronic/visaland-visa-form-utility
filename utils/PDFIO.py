__all__ = ['PDFIO', 'XFAPDF', 'CanadaXFA']

# import packages
from types import FunctionType
from enum import Enum
import os
import logging
from fnmatch import fnmatch

# preprocessing
import re
from typing import Callable
import xml.etree.ElementTree as et
import xmltodict
from collections import OrderedDict

# PDF tools
import pikepdf
import PyPDF2 as pypdf

# our modules
from utils.constant import DOC_TYPES
from utils.helpers import deprecated, loggingdecorator

# logging
logger = logging.getLogger('__main__')


class PDFIO:
    """
    Base class for dealing with PDF files
    """

    def __init__(self) -> None:
        pass

    def extract_raw_content(self, pdf_path: str) -> str:
        """
        Extracts unprocessed data from a PDF file

        args:
            pdf_path: Path to the pdf file
        """

        raise NotImplementedError

    def find_in_dict(self, needle, haystack):
        for key in haystack.keys():
            try:
                value = haystack[key]
            except:
                continue
            if key == needle:
                return value
            if isinstance(value, dict):
                x = self.find_in_dict(needle, value)
                if x is not None:
                    return x


class XFAPDF(PDFIO):
    """
    Contains functions and utility tools for dealing with XFA PDF documents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(logger.name+'.XFAPDF')

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, input=True, output=True)
    def make_machine_readable(self, src: str, dst: str) -> None:
        """
        Method that reads a 'content-copy' protected PDF and removes this restriction
        by saving a "printed" version of.

        Ref: https://www.reddit.com/r/Python/comments/t32z2o/simple_code_to_unlock_all_readonly_pdfs_in/

        args:
            file_name: file name, if None, considers all files in `src_path`
            src: source file path
            dst: destination (processed) file path 
        """

        pdf = pikepdf.open(src, allow_overwriting_input=True)
        pdf.save(dst)

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.INFO, input=True, output=True)
    def process_directory(self, src_dir: str, dst_dir: str,
                          func: Callable, pattern: str = '*'):
        """
        Iterate through `src_dir`, processing all files that match pattern via
            given function `func` (for single file) and store them,
            including their parent directories in `dst_dir`.

        args:
            src_dir: Source directory to be processed
            dst_dir: Destination directory to write processed files
            pattern: pattern to match files, default to '*' for all files

        ref: https://stackoverflow.com/a/24041933/18971263
        """

        assert src_dir != dst_dir, 'Source and destination dir must differ.'
        if src_dir[-1] != '/':
            src_dir += '/'
        for dirpath, dirnames, all_filenames in os.walk(src_dir):
            # filter out files that match pattern only
            filenames = filter(lambda fname: fnmatch(
                fname, pattern), all_filenames)

            if filenames:
                dir_ = os.path.join(dst_dir, dirpath.replace(src_dir, ''))
                os.makedirs(dir_, exist_ok=True)
                for fname in filenames:
                    in_fname = os.path.join(dirpath, fname)
                    out_fname = os.path.join(dir_, fname)
                    func(in_fname, out_fname)

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def extract_raw_content(self, pdf_path: str) -> str:
        """
        Extracts RAW content of XFA PDF files which are in XML.

        Ref: https://towardsdatascience.com/how-to-extract-data-from-pdf-forms-using-python-10b5e5f26f70
        """

        pdfobject = open(pdf_path, 'rb')
        pdf = pypdf.PdfFileReader(pdfobject)
        xfa = self.find_in_dict('/XFA', pdf.resolvedObjects)
        # `datasets` keyword contains filled forms in XFA array
        xml = xfa[xfa.index('datasets')+1].getObject().getData()
        xml = str(xml)  # convert bytes to str
        return xml

    def clean_xml_for_csv(self, xml: str, type: Enum) -> str:
        """
        Cleans the XML file extracted from XFA forms
        Remark: since each form has its own format and issues, this method needs
            to be implemented uniquely for each unique file/form which needs
            to be specified using argument `type` that can be populated from `DOC_TYPES`.

        args:
            xml: A string containing XML code
            str: The type of XFA form (e.g. 5257e, 5645e,)
        """

        raise NotImplementedError

    @deprecated('Use `flatten_dict`')
    def flatten_dict_basic(self, d: dict) -> dict:
        """
        Takes a (nested) dictionary and flattens it where the final keys are key.key....
            and values are the leaf values of dictionary.

        ref: https://stackoverflow.com/questions/38852822/how-to-flatten-xml-file-in-python
        args:
            d: A dictionary  
            return: An ordered dict
        """

        def items():
            for key, value in d.items():
                if isinstance(value, dict):
                    for subkey, subvalue in self.flatten_dict_basic(value).items():
                        yield key + "." + subkey, subvalue
                else:
                    yield key, value

        return dict(items())

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def flatten_dict(self, d: dict) -> dict:
        """
        Takes a (nested) multilevel dictionary and flattens it where the final keys are key.key....
            and values are the leaf values of dictionary.

        ref: https://stackoverflow.com/a/67744709/18971263
        args:
            d: A dictionary  
            return: An ordered dict
        """

        def items():
            if isinstance(d, dict):
                for key, value in d.items():
                    # nested subtree
                    if isinstance(value, dict):
                        for subkey, subvalue in self.flatten_dict(value).items():
                            yield '{}.{}'.format(key, subkey), subvalue
                    # nested list
                    elif isinstance(value, list):
                        for num, elem in enumerate(value):
                            for subkey, subvalue in self.flatten_dict(elem).items():
                                yield '{}.[{}].{}'.format(key, num, subkey), subvalue
                    # everything else (only leafs should remain)
                    else:
                        yield key, value
        return dict(items())

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def xml_to_flattened_dict(self, xml: str) -> dict:
        """
        Takes a (nested) XML and flattens it to a dict where the final keys are key.key....
            and values are the leaf values of XML tree.

        args:
            d: A XML string
            return: A flattened ordered dict
        """
        flattened_dict = xmltodict.parse(xml)  # XML to dict
        flattened_dict = self.flatten_dict(flattened_dict)
        return flattened_dict


class CanadaXFA(XFAPDF):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(logger.name+'.CanadaXFA')

    @loggingdecorator(logger.name+'.CanadaXFA.func', level=logging.DEBUG, output=False)
    def clean_xml_for_csv(self, xml: str, type: Enum) -> str:
        if type == DOC_TYPES.canada_5257e:
            # remove bad characters
            xml = re.sub(r"b'\\n", '', xml)
            xml = re.sub(r"'", '', xml)
            xml = re.sub(r"\\n", '', xml)

            # remove 9000 lines of redundant info for '5257e' doc
            tree = et.ElementTree(et.fromstring(xml))
            root = tree.getroot()
            junk = tree.findall('LOVFile')
            root.remove(junk[0])
            xml = str(et.tostring(root, encoding='utf8', method='xml'))
            # parsing through ElementTree adds bad characters too
            xml = re.sub(
                r"b'<\?xml version=\\'1.0\\' encoding=\\'utf8\\'\?>", '', xml)
            xml = re.sub(r"'", '', xml)
            xml = re.sub(r"\\n[ ]*", '', xml)

        elif type == DOC_TYPES.canada_5645e:
            # remove bad characters
            xml = re.sub(r"b'\\n", '', xml)
            xml = re.sub(r"'", '', xml)
            xml = re.sub(r"\\n", '', xml)

        return xml
