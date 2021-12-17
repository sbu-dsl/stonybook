from pathlib import Path
import lxml.etree as ET
from unidecode import unidecode
import re

from stonybook.preprocessing.gutenberg.gutenberg_util import get_strip_locations


def valid_xml_char_ordinal(c):
    codepoint = ord(c)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF or
        codepoint in (0x9, 0xA, 0xD) or
        0xE000 <= codepoint <= 0xFFFD or
        0x10000 <= codepoint <= 0x10FFFF
        )

def get_initial_xml(lines, start_line, end_line):
    sep = '\n'
    book = ET.Element("book")
    meta = ET.SubElement(book, "meta")
    gutenberg_header = ET.SubElement(meta, "gutenberg_header")
    gutenberg_footer = ET.SubElement(meta, "gutenberg_footer")
    body = ET.SubElement(book, "body")
    
    gutenberg_header.text = ''.join(c for c in sep.join(lines[:start_line]) if valid_xml_char_ordinal(c))
    body.text = ''.join(c for c in sep.join(lines[start_line:end_line]) if valid_xml_char_ordinal(c))
    gutenberg_footer.text = ''.join(c for c in sep.join(lines[end_line:]) if valid_xml_char_ordinal(c))
    
    return book


def get_clean_xml(book):
    body = book.find('.//body')
    clean_contents = unidecode(body.text)
    body.text = clean_contents
    
    return book



def add_tags(body, start_end_list):
    
    if len(start_end_list) == 0:
        return body
    
    text = body.text
    
    start_0, end_0, tag_name_0, strip_char_0 = start_end_list[0]
    
    body.text = text[:start_0]
    
    for i, elem in enumerate(start_end_list[:-1]):
        start, end, tag_name, strip_char = elem
        next_start = start_end_list[i + 1][0]
        new_elem = ET.SubElement(body, tag_name)
        new_elem.text = text[start:end].strip(strip_char)
        new_elem.tail = text[end:next_start]
        
    start_1, end_1, tag_name_1, strip_char_1 = start_end_list[-1]
    new_elem = ET.SubElement(body, tag_name_1)
    new_elem.text = text[start_1:end_1].strip(strip_char_1)
    new_elem.tail = text[end_1:]
    
    return body


def get_span_list_from_rule(tag_name, rule_text, body, strip_char=''):
    r = re.compile(rule_text)
    
    l = list()
    for m in r.finditer(body.text):
#         print(m)
        l.append(list(m.span()) + [tag_name, strip_char])
    
    return l
    

def annotate_all(book):
    
    body = book.find('.//body')
    
    if body.text is None:
        return book
    
    d = dict()
    
    d['italic'] = '(?<=\s)_[^\s][^_]*[^\s]_(?=\s)'
    d['bold'] = '(?<=\s)\=[^\s][^\=]*[^\s]\=(?!\=)'
    d['star_sep'] = '(?<=\n)[^\S\r\n]*[\*][\*\s]*(?=\n)'
    d['illustration'] = '\[Illustration[^\]]*\]'
    d['transcriber_note'] = '\[Transcriber[^\]]*\]'
    
    body.text = '\n' + body.text
    
    s = set()
    l = list()
    
    for elem in ['transcriber_note', 'illustration', 'italic', 'bold', 'star_sep']:
    
#     for elem in ['transcriber_note']:
        
        if elem == 'italic':
            strip_char = '_'
        elif elem == 'bold':
            strip_char = '='
        else:
            strip_char = ''
    
        l2 = get_span_list_from_rule(elem, d[elem], body, strip_char)

        for elem in l2:
            if any([i in s for i in range(elem[0], elem[1])]):
                continue
            for i in range(elem[0], elem[1]):
                s.add(i)
            l.append(elem)
    
    body = add_tags(body, sorted(l))
    
    return book
    

def save_to_file(book, filename):
    filename = str(filename)
    tree = ET.ElementTree(book)
    tree.write(filename, pretty_print=True, encoding='utf-8')
    return
    
def convert_raw_to_base(book):
    
    ET.strip_tags(book, 'italic')
    ET.strip_tags(book, 'bold')
    
    ET.strip_elements(book, 'illustration', with_tail=False)
    ET.strip_elements(book, 'transcriber_note', with_tail=False)
    
    body = book.find('.//body')
    children = body.getchildren()
    
    if body.text is not None:
        paras = re.split('\n\n\s*', body.text)
        paras = [x for x in paras if len(x.strip()) > 0]
        body.text = None
        for i, p in enumerate(paras):
            new_elem = ET.Element('p')
            new_elem.text = p
            body.insert(i, new_elem)
    
    for c in children:
        paras = re.split('\n\n\s*', c.tail)
        paras = [x for x in paras if len(x.strip()) > 0]
        c.tail = None
        prev_tag = c
        for p in paras:
            new_elem = ET.Element('p')
            new_elem.text = p
            prev_tag.addnext(new_elem)
            prev_tag = new_elem
            
    ET.strip_elements(book, 'gutenberg_header')
    ET.strip_elements(book, 'gutenberg_footer')
    
    return book
    
def gutenberg_preprocess(input_txt_file, output_dir, force_raw=False, force_base=False):
    # input_txt_file:  <book_id>.txt
    # Outputs:         output_dir/book_id/raw.xml, output_dir/book_id/base.xml
    
    input_txt_file = Path(input_txt_file)
    output_dir = Path(output_dir)
    
    book_id = input_txt_file.stem
    output_dir = output_dir / book_id
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    output_path_raw = output_dir/"raw.xml"
    output_path_base = output_dir/"base.xml"
    
    if force_raw or (not output_path_raw.exists()):
    
        with open(input_txt_file, 'r') as f:
            content = f.read()

        start_line, end_line = get_strip_locations(content)

        lines = content.splitlines()

        initial_xml = get_initial_xml(lines, start_line, end_line)

        clean_xml = get_clean_xml(initial_xml)

        processed_xml = annotate_all(clean_xml)

        save_to_file(processed_xml, output_path_raw)
    
    if force_base or (not output_path_base.exists()):
        # Read raw file
        # Convert to base (remove annotation tags, add paragraph tags)

        parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
        tree = ET.parse(str(output_path_raw), parser=parser)
        book = tree.getroot()

        book = convert_raw_to_base(book)
        save_to_file(book, output_path_base)
    
    
    
    