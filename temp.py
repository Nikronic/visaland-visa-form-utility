
from utils import XFAPDF

xml_file_path = 'sample/xml/5645e-17.xml'

src_path = 'sample/enc_sample/'  # path to source encrypted pdf
des_path = 'sample/dec_sample/'  # path to decrypted pdf

SAMPLE = 'sample/dec_sample/imm5257e.pdf'

xfa_pdf = XFAPDF()

xfa_pdf.make_machine_readable(file_name=None, src_path=src_path, des_path=des_path)
xml = xfa_pdf.extract_raw_content(SAMPLE)

# remove bad characters
import re
xml_cleaned = re.sub(r"b'\\n", '', xml)
xml_cleaned = re.sub(r"'", '', xml_cleaned)
xml_cleaned = re.sub(r"\\n", '', xml_cleaned)

# remove 9000 lines of redundant info for '5257e' doc
import xml.etree.ElementTree as et
tree = et.ElementTree()
tree.parse('sample/xml/5257e-9-cleaned.xml')
root = tree.getroot()
junk = tree.findall('LOVFile')  
root.remove(junk[0])
xml_cleaned2 = str(et.tostring(root, encoding='utf8', method='xml'))
xml_cleaned2 = re.sub(r"b'<\?xml version=\\'1.0\\' encoding=\\'utf8\\'\?>", '', xml_cleaned2)
xml_cleaned2 = re.sub(r"'", '', xml_cleaned2)
xml_cleaned2 = re.sub(r"\\n[ ]*", '', xml_cleaned2)

print
