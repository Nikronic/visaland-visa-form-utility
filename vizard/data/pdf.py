__all__ = [
    'PDFIO', 'XFAPDF', 'CanadaXFA'
]

# core
import xml.etree.ElementTree as et
import PyPDF2 as pypdf
import re
# ours: data
from vizard.data import functional
from vizard.data.constant import DOC_TYPES
# ours: helpers
from vizard.utils.helpers import deprecated
from vizard.utils.helpers import loggingdecorator
# helpers
from enum import Enum
from typing import Any
import logging


# logging
logger = logging.getLogger(__name__)


class PDFIO:
    """Base class for dealing with PDF files

    For each type of PDF, let's say XFA files, one needs to extend
    this class and abstract methods like :func:`extract_raw_content`
    to generate a string of the content of the PDF in a format that
    can be used by the other classes (e.g. `XML`). For instance,
    see :class:`XFAPDF` for the extension of this class.

    """

    def __init__(self) -> None:
        pass

    def extract_raw_content(self, pdf_path: str) -> str:
        """Extracts unprocessed data from a PDF file

        Args:
            pdf_path (str): Path to the pdf file
        """

        raise NotImplementedError

    def find_in_dict(self, needle: Any, haystack: Any) -> Any:
        """Looks for the value of a key inside a nested dictionary

        Args:
            needle (Any): Key to look for
            haystack (Any): Dictionary to look in. Can be a dict inside
                another dict

        Returns:
            Any: The value of key ``needle``
        """
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
    """Contains functions and utility tools for dealing with XFA PDF documents.

    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(logger.name+'.XFAPDF')

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def extract_raw_content(self, pdf_path: str) -> str:
        """Extracts RAW content of XFA PDF files which are in XML format

        Args:
            pdf_path (str): path to the pdf file

        Reference:

            * https://towardsdatascience.com/how-to-extract-data-from-pdf-forms-using-python-10b5e5f26f70


        Returns:
            str: XFA object of the pdf file in XML format
        """

        pdfobject = open(pdf_path, 'rb')
        pdf = pypdf.PdfFileReader(pdfobject)
        xfa = self.find_in_dict('/XFA', pdf.resolved_objects)
        # `datasets` keyword contains filled forms in XFA array
        xml = xfa[xfa.index('datasets')+1].get_object().get_data()
        xml = str(xml)  # convert bytes to str
        return xml

    def clean_xml_for_csv(self, xml: str, type: Enum) -> str:
        """Cleans the XML file extracted from XFA forms

        Since each form has its own format and issues, this method needs
        to be implemented uniquely for each unique file/form which needs
        to be specified using argument ``type`` that can be populated from
        ``DOC_TYPES``.

        Args:
            xml (str): XML content
            type (Enum): type of the document defined
                in :class:`DOC_TYPES <vizard.data.constant.DOC_TYPES>`

        Returns:
            str: cleaned XML content to be used in CSV file
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
        """Takes a (nested) multilevel dictionary and flattens it

        The final keys are ``key.key...`` and values are the leaf values of dictionary

        Args:
            d (dict): A dictionary

        References:

            * https://stackoverflow.com/a/67744709/18971263

        Returns:
            dict: A flattened dictionary
        """
        return functional.flatten_dict(d=d)

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def xml_to_flattened_dict(self, xml: str) -> dict:
        """Takes a (nested) XML and converts it to a flattened dictionary

        The final keys are ``key.key...`` and values are the leaf values of XML tree

        Args:
            xml (str): A XML string

        Returns:
            dict: A flattened dictionary
        """
        return functional.xml_to_flattened_dict(xml=xml)


class CanadaXFA(XFAPDF):
    """Handles Canada XFA PDF files

    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(logger.name+'.CanadaXFA')

    @loggingdecorator(logger.name+'.CanadaXFA.func', level=logging.DEBUG, output=False)
    def clean_xml_for_csv(self, xml: str, type: Enum) -> str:
        """Hardcoded cleaning of Canada XFA XML files to be XML compatible with CSV

        Args:
            xml (str): XML content
            type (Enum): type of the document defined
                in :class:`DOC_TYPES <vizard.data.constant.DOC_TYPES>`

        Returns:
            str: cleaned XML content to be used in CSV file
        """
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
