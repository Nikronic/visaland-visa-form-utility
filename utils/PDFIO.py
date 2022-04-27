#### 
# Contains implementation related to dealing with PDF, mostly IO
# Author: Nikan Doosti @ NahalGasht
####

# TODO: add logging

# import packages
from typing import Union
from enum import Enum
import os
from pickle import DICT

# preprocessing
import re
import xml.etree.ElementTree as et

# PDF tools
import pikepdf
import PyPDF2 as pypdf



class DOC_TYPES(Enum):
    """
    Contains all document types which can be used to customize ETL steps for each doc.
    Remark: Order of docs is meaningless.
    """
    
    canada_5257e = 1  # application for visitor visa (temporary resident visa)
    canada_5645e = 2  # Family information


class PDFIO:
    """
    Base class for dealing with PDF files
    """
    def __init__(self) -> None:
        pass

    def extract_raw_content(pdf_path: str) -> None:
        """
        Extracts unprocessed data from a PDF file

        args:
            pdf_path: Path to the pdf file
        """

        raise NotImplementedError

    def find_in_dict(self, needle, haystack):
        for key in haystack.keys():
            try:
                value=haystack[key]
            except:
                continue
            if key==needle:
                return value
            if isinstance(value,dict):            
                x=self.find_in_dict(needle,value)            
                if x is not None:
                    return x



class XFAPDF(PDFIO):
    """
    Contains functions and utility tools for dealing with XFA PDF documents.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def make_machine_readable(self, file_name: Union[str, None], src_path: str, des_path: str) -> None:
        """
        Method that reads a 'content-copy' protected PDF and removes this restriction
        by saving a "printed" version of.

        Ref: https://www.reddit.com/r/Python/comments/t32z2o/simple_code_to_unlock_all_readonly_pdfs_in/

        args:
            file_name: file name, if None, considers all files in `src_path`
            src_path: directory address of files
            des_path: destination directory address 
        """

        if file_name is None:
            files = [f for f in os.listdir(src_path)]
            for file_name in files:
                pdf = pikepdf.open(src_path+file_name, allow_overwriting_input=True)
                pdf.save(des_path+file_name)
        return None

    def extract_raw_content(self, pdf_path: str) -> str:
        """
        Extracts RAW content of XFA PDF files which are in XML.

        Ref: https://towardsdatascience.com/how-to-extract-data-from-pdf-forms-using-python-10b5e5f26f70
        """

        pdfobject=open(pdf_path,'rb')
        pdf=pypdf.PdfFileReader(pdfobject)
        xfa=self.find_in_dict('/XFA',pdf.resolvedObjects)
        xml=xfa[xfa.index('datasets')+1].getObject().getData() # `datasets` keyword contains filled forms in XFA array
        xml = str(xml)  # convert bytes to str
        return xml

    def clean_xml_for_csv(self, xml: str, type: str) -> str:
        """
        Cleans the XML file extracted from XFA forms
        Remark: since each form has its own format and issues, this method needs
            to be implemented uniquely for each unique file/form which needs
            to be specified using argument `type`.
        
        args:
            xml: A string containing XML code
            str: The type of XFA form (e.g. 5257e, 5645e,)
        """

        raise NotImplementedError

class CanadaXFA(XFAPDF):
    def __init__(self) -> None:
        super().__init__()
    
    def clean_xml_for_csv(self, xml: str, type: Enum) -> str:
        if type == DOC_TYPES.canada_5257e:
            # remove bad characters
            xml = re.sub(r"b'\\n", '', xml)
            xml = re.sub(r"'", '', xml)
            xml = re.sub(r"\\n", '', xml)

            # remove 9000 lines of redundant info for '5257e' doc
            tree = et.ElementTree()
            tree.parse('sample/xml/5257e-9-cleaned.xml')
            root = tree.getroot()
            junk = tree.findall('LOVFile')  
            root.remove(junk[0])
            xml = str(et.tostring(root, encoding='utf8', method='xml'))
            # parsing through ElementTree adds bad characters too
            xml = re.sub(r"b'<\?xml version=\\'1.0\\' encoding=\\'utf8\\'\?>", '', xml)
            xml = re.sub(r"'", '', xml)
            xml = re.sub(r"\\n[ ]*", '', xml)
        
        elif DOC_TYPES.canada_5645e:
            # remove bad characters
            xml = re.sub(r"b'\\n", '', xml)
            xml = re.sub(r"'", '', xml)
            xml = re.sub(r"\\n", '', xml)
        
