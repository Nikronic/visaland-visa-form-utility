from xml.etree import cElementTree as ET
from utils import XFAPDF

xml_file_path = 'sample/xml/5645e-17.xml'

# src_path = 'sample/enc_sample/'  # path to source encrypted pdf
# des_path = 'sample/dec_sample/'  # path to decrypted pdf

# SAMPLE = 'sample/dec_sample/imm5645e.pdf'

# xfa_pdf = XFAPDF()

# xfa_pdf.make_machine_readable(file_name=None, src_path=src_path, des_path=des_path)
# xml = xfa_pdf.extract_raw_content(SAMPLE)

# remove bad characters


for event, elem in ET.iterparse(xml_file_path, events=("start", "end")):
    if event == 'end' and elem.tag == record_category and elem.attrib['action'] != 'del':
        record_contents = get_record(elem, name_types=name_types, name_components=name_components, record_id=elem.attrib['id'])

