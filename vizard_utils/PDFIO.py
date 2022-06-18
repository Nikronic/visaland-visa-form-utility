__all__ = ['PDFIO', 'XFAPDF', 'CanadaXFA']

# import packages
from enum import Enum
import logging

# preprocessing
import re
import xml.etree.ElementTree as et

# PDF tools
import PyPDF2 as pypdf

# our modules
from vizard_utils import functional
from vizard_utils.constant import DOC_TYPES
from vizard_utils.helpers import deprecated, loggingdecorator

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

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def extract_raw_content(self, pdf_path: str) -> str:
        """
        Extracts RAW content of XFA PDF files which are in XML.

        Ref: https://towardsdatascience.com/how-to-extract-data-from-pdf-forms-using-python-10b5e5f26f70
        """

        pdfobject = open(pdf_path, 'rb')
        pdf = pypdf.PdfFileReader(pdfobject)
        xfa = self.find_in_dict('/XFA', pdf.resolved_objects)
        # `datasets` keyword contains filled forms in XFA array
        xml = xfa[xfa.index('datasets')+1].get_object().get_data()
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
        return functional.flatten_dict(d=d)

    @loggingdecorator(logger.name+'.XFAPDF.func', level=logging.DEBUG, output=False)
    def xml_to_flattened_dict(self, xml: str) -> dict:
        """
        Takes a (nested) XML and flattens it to a dict where the final keys are key.key....
            and values are the leaf values of XML tree.

        args:
            xml: A XML string
        """
        return functional.xml_to_flattened_dict(xml=xml)


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
