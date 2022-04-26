#### 
# Contains implementation related to dealing with PDF, mostly IO
# Author: Nikan Doosti @ NahalGasht
####

# TODO: add logging

# import packages
from typing import Union
import os
from pickle import DICT

# PDF tools
import pikepdf
import PyPDF2 as pypdf


class PDFIO:
    def __init__(self) -> None:
        pass

    def extract_raw_content(pdf_path: str) -> None:
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
        return xml

