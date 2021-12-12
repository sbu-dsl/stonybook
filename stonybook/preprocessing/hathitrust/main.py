from pathlib import Path
import zipfile
import os
from lxml import etree
import copy
from nltk.corpus import words
from string import ascii_lowercase
from unidecode import unidecode
from collections import Counter

from stonybook.preprocessing.hathitrust.clean_hathi_headers import htrc_cleaned_pages

import sys,re


def parse_grouped_sections(unzipped_loc):
    
    from stonybook.preprocessing.hathitrust.book_segmentation.code import segment_from_feature_file
    from stonybook.preprocessing.hathitrust.book_segmentation.code import featurize_book
    
    modelFolder = '/nfs/nfs-davinci/cpethe/books/Sem9/stonybook/stonybook/preprocessing/hathitrust/book_segmentation/models/labseg10'
    
    meanFile="%s/means.txt" % modelFolder
    modelFile="%s/new_model.ckpt" % modelFolder
    vocabFile="%s/vocab.txt" % modelFolder
    
    featurize_book.readVocab(vocabFile)
    segment_from_feature_file.readmeans(meanFile)

    book=featurize_book.book(str(unzipped_loc))
    feats, page_index=segment_from_feature_file.convertBookToFeats(book.pages)

    predictions=segment_from_feature_file.predict(feats, page_index, modelFile)
    
    lines = [list(x) for x in predictions]
    
    collapsed_sections = []
    if not lines:
        return [[],[],[]]
    page_start = int(lines[0][0])
    curr_lab = lines[0][1]
    for page_num, lab in lines:
        if lab == curr_lab:
            continue
        collapsed_sections.append((page_start, int(page_num), curr_lab))
        curr_lab = lab
        page_start = int(page_num)
    collapsed_sections.append((page_start, int(page_num)+1, curr_lab))
    max_content_length = -1
    max_content_idx = -1
    for i, section in enumerate(collapsed_sections):
        start, end, label = section
        if label == 'content':
            content_length = end - start
            if content_length > max_content_length:
                max_content_length = content_length
                max_content_idx = i

    grouped_sections = [
        collapsed_sections[:max_content_idx],
        [collapsed_sections[max_content_idx]],
        collapsed_sections[max_content_idx+1:]
    ]
    return grouped_sections




def generate_xml_tree(grouped_sections, parsed_pages, input_dir, output_path_raw):
    book_pages = {}
    for i, page_name in enumerate(sorted(os.listdir(input_dir))):
        book_num = page_name.split(".")[0][-8:]
        if '_' in book_num:
            book_num = book_num.split('_')[-1]
        ignore = ['notes', 'pagedata']
        if book_num in ignore:
            continue
        book_num = int(book_num)
        book_pages[book_num] = parsed_pages[i]

    root = etree.Element("book")
    meta = etree.SubElement(root, "meta")
    front = etree.SubElement(root, "front")
    body = etree.SubElement(root, "body")
    back = etree.SubElement(root, "back")

    possible_labels = [
        "title",
        "advertisement",
        "publisher",
        "dedication",
        "preface",
        "toc",
        "appendix",
        "index"
    ]

    def fill_section(book_section, book_section_range):
        for start, end, label in book_section_range:
            for i in range(start, end):
                if i not in book_pages:
                    continue
                page = etree.SubElement(book_section, "page")
                page.attrib["num"] = str(i)
                
                # Language detection
                content = '\n'.join(book_pages[i])
                try:
                    lang = detect(content)
                except:
                    lang = None
                
                page.attrib["lang"] = str(lang)
                
                if label in possible_labels:
                    page.attrib["bamman_tag"] = label
                # j loops over header,body,footer
                tags = [
                    "page_header",
                    "page_body",
                    "page_footer"
                ]
                for j, tag in enumerate(tags):
                    page_tag = etree.SubElement(page, tag)
                    page_tag.text = book_pages[i][j]

    for i, section in enumerate([front, body, back]):
        fill_section(section, grouped_sections[i])

    book = etree.ElementTree(root)
    book.write(str(output_path_raw), pretty_print=True, encoding='utf-8')
    return True



def most_common(arr):
    c = Counter(arr)
    return c.most_common(1)[0][0]


def strip_numerals(header, body, footer):
    if not body:
        return (header, body, footer)
    body_lines = body.split('\n')
    while body_lines and body_lines[0].strip().isnumeric():
        header = header + '\n' + body_lines[0]
        body_lines = body_lines[1:]
    while body_lines and body_lines[-1].strip().isnumeric():
        footer = body_lines[-1] + '\n' + footer
        body_lines = body_lines[:-1]
    return (header, '\n'.join(body_lines), footer)

def strip_nonalpha(header, body, footer):
    if not body:
        return (header, body, footer)
    body_lines = body.split('\n')
    while body_lines and not any(c.isalpha() for c in body_lines[0]):
        header += '\n' + body_lines[0]
        body_lines = body_lines[1:]
    while body_lines and not any(c.isalpha() for c in body_lines[-1]):
        footer = body_lines[-1] + '\n' + footer
        body_lines = body_lines[:-1]
    return (header, '\n'.join(body_lines), footer)

def strip_start_end_numerals(header, body, footer):
    if not body:
        return (header, body, footer)
    body_lines = body.split('\n')
    while body_lines and body_lines[0].strip() and (any(c.isnumeric() for c in body_lines[0].split()[0])
        or any(c.isnumeric() for c in body_lines[0].split()[-1])):
        header += '\n' + body_lines[0]
        body_lines = body_lines[1:]
    while body_lines and body_lines[-1].strip() and (any(c.isnumeric() for c in body_lines[-1].split()[0])
        or any(c.isnumeric() for c in body_lines[-1].split()[-1])):
        footer = body_lines[-1] + '\n' + footer
        body_lines = body_lines[:-1]
    return (header, '\n'.join(body_lines), footer)

def strip_upper_title(header, body, footer):
    if not body:
        return (header, body, footer)
    body_lines = body.split('\n')
    while body_lines and (body_lines[0].istitle() or body_lines[0].isupper()):
        header += '\n' + body_lines[0]
        body_lines = body_lines[1:]
    return (header, '\n'.join(body_lines), footer)

def strip_numerals(header, body, footer):
    if not body:
        return (header, body, footer)
    body_lines = body.split('\n')
    while body_lines and not any(c.isalpha() for c in body_lines[0]):
        header += '\n' + body_lines[0]
        body_lines = body_lines[1:]
    while body_lines and not any(c.isalpha() for c in body_lines[-1]):
        footer = body_lines[-1] + '\n' + footer
        body_lines = body_lines[:-1]
    return (header, '\n'.join(body_lines), footer)

def normalize_tok(tok):
    return re.sub(r'[.\'\";,`’“”\-\!\?]', '', tok).lower()

def strip_num_header_lines(header, body, footer, num):
    if not body or num <= 0:
        return (header, body, footer)
    body_lines = body.split('\n')
    while body_lines and num > 0:
        header += '\n' + body_lines[0]
        body_lines = body_lines[1:]
        num -= 1
    return (header, '\n'.join(body_lines), footer)

def strip_final(header, body, footer, english_words):
    if not body:
        return (header, body, footer)

    body_lines = body.split('\n')
    while body_lines:
        if not body_lines[0] or body_lines[0].isspace():
            header += '\n' + body_lines[0]
            body_lines = body_lines[1:]
            continue

        first_line_toks = body_lines[0].split()
        first_tok = first_line_toks[0]
        last_tok = first_line_toks[-1]
        num_weird = sum([normalize_tok(t) not in english_words for t in first_line_toks])
        if (any(c.isnumeric() for c in first_tok)
                or any(c.isnumeric() for c in last_tok)
                or num_weird / len(first_line_toks) > 0.75):
            header += '\n' + body_lines[0]
            body_lines = body_lines[1:]
        else:
            break
    while body_lines:
        if not body_lines[-1] or body_lines[-1].isspace():
            footer = body_lines[-1] + '\n' + footer
            body_lines = body_lines[:-1]
            continue
        last_line_toks = body_lines[-1].split()
        first_tok = last_line_toks[0]
        last_tok = last_line_toks[-1]
        # edge case when last line is last word of paragraph like
        # Bob was never em-
        # ployed.
        if len(last_line_toks) == 1:
            num_weird = sum([normalize_tok(t) not in english_words for t in last_line_toks])
            if first_tok[-1] in ".!?" and normalize_tok(first_tok).isalpha():
                break
        num_weird = sum([normalize_tok(t) not in english_words for t in last_line_toks])
        if (any(c.isnumeric() for c in first_tok)
                or any(c.isnumeric() for c in last_tok)
                or num_weird / len(last_line_toks) > 0.75):
            footer = body_lines[-1] + '\n' + footer
            body_lines = body_lines[:-1]
        else:
            break
    return (header, '\n'.join(body_lines), footer)




def filter_xml(book, strip_foo):
    body = book.find('body')
    for page in body:
        page_header = ""
        page_body = ""
        page_footer = ""
        for child in page:
            if child.text == None:
                child.text = ""
            if child.tag == 'page_header':
                page_header = child.text.strip()
            elif child.tag == 'page_body':
                page_body = child.text.strip()
            elif child.tag == 'page_footer':
                page_footer = child.text.strip()
        new_page = strip_foo(page_header, page_body, page_footer)
        for child in page:
            child.getparent().remove(child)
        tag_names = [
            "page_header",
            "page_body",
            "page_footer"
        ]
        for i, tag_name in enumerate(tag_names):
            tag = etree.SubElement(page, tag_name)
            tag.text = new_page[i].strip()
    return book

def filter_majority_xml(book, strip_foo, ratio=0.5):
    book_copy = copy.deepcopy(book)
    body = book.find('body')
    num_page_changed = 0
    total_num_page = 0
    for page in body:
        total_num_page += 1
        page_header = ""
        page_body = ""
        page_footer = ""
        for child in page:
            if child.text == None:
                child.text = ""
            if child.tag == 'page_header':
                page_header = child.text.strip()
            elif child.tag == 'page_body':
                page_body = child.text.strip()
            elif child.tag == 'page_footer':
                page_footer = child.text.strip()
        new_page = strip_foo(page_header, page_body, page_footer)
        if new_page[1].strip() != page_body.strip():
            num_page_changed += 1
        for child in page:
            child.getparent().remove(child)
        tag_names = [
            "page_header",
            "page_body",
            "page_footer"
        ]
        for i, tag_name in enumerate(tag_names):
            tag = etree.SubElement(page, tag_name)
            tag.text = new_page[i].strip()
    if total_num_page == 0 or num_page_changed / total_num_page < ratio:
        return book_copy
    return book

def find_upper_title_numbers(body):
    if not body:
        return 0
    body_lines = body.split('\n')
    idx = 0
    num_lines = 0
    for idx, line in enumerate(body_lines):
        if line.istitle() or line.isupper() or line.isnumeric():
            num_lines += 1
        else:
            break
    return num_lines

def filter_final(book):
    body = book.find('body')
    english_words = set(words.words())
    for letter in ascii_lowercase:
        if letter == 'a' or letter == 'i':
            continue
        english_words.remove(letter)
    for page in body:
        page_header = ""
        page_body = ""
        page_footer = ""
        for child in page:
            if child.text == None:
                child.text = ""
            if child.tag == 'page_header':
                page_header = child.text.strip()
            elif child.tag == 'page_body':
                page_body = child.text.strip()
            elif child.tag == 'page_footer':
                page_footer = child.text.strip()
        new_page = strip_final(page_header, page_body, page_footer, english_words)
        for child in page:
            child.getparent().remove(child)
        tag_names = [
            "page_header",
            "page_body",
            "page_footer"
        ]
        for i, tag_name in enumerate(tag_names):
            tag = etree.SubElement(page, tag_name)
            tag.text = new_page[i].strip()

    num_page_changed = 0
    total_num_page = 0
    body_lines_changed = []
    all_lines_changed = []
    for page in body:
        total_num_page += 1
        page_header = ""
        page_body = ""
        page_footer = ""
        for child in page:
            if child.text == None:
                child.text = ""
            if child.tag == 'page_header':
                page_header = child.text.strip()
            elif child.tag == 'page_body':
                page_body = child.text.strip()
            elif child.tag == 'page_footer':
                page_footer = child.text.strip()
        num_lines_changed = find_upper_title_numbers(page_body)
        if num_lines_changed > 0:
            num_page_changed += 1
        body_lines_changed.append(num_lines_changed)
        num_header_lines = 0
        if page_header != "":
            num_header_lines = len(page_header.split('\n'))
        all_lines_changed.append(num_lines_changed + num_header_lines)

    if total_num_page > 0 and num_page_changed / total_num_page > 0.25:
        lines_to_strip = most_common(all_lines_changed)
        if lines_to_strip > 0:
            for i, page in enumerate(body):
                page_header = ""
                page_body = ""
                page_footer = ""
                for child in page:
                    if child.text == None:
                        child.text = ""
                    if child.tag == 'page_header':
                        page_header = child.text.strip()
                    elif child.tag == 'page_body':
                        page_body = child.text.strip()
                    elif child.tag == 'page_footer':
                        page_footer = child.text.strip()
                num_header_lines = 0
                if page_header != "":
                    num_header_lines = len(page_header.split('\n'))
                page_lines_to_strip = min(body_lines_changed[i], lines_to_strip - num_header_lines)
                new_page = strip_num_header_lines(page_header, page_body, page_footer, page_lines_to_strip)
     
                for child in page:
                    child.getparent().remove(child)
                tag_names = [
                    "page_header",
                    "page_body",
                    "page_footer"
                ]
                for j, tag_name in enumerate(tag_names):
                    tag = etree.SubElement(page, tag_name)
                    tag.text = new_page[j].strip()
     
    return book



def clean_header_footers(book):
    book = filter_xml(book, strip_numerals)
    book = filter_xml(book, strip_nonalpha)
    book = filter_xml(book, strip_start_end_numerals)
    book = filter_majority_xml(book, strip_upper_title, ratio=0.5)
    book = filter_final(book)
    return book


def merge_hyphens(text):
    lines = text.splitlines(keepends=True)
    for num in range(len(lines) - 1):
        # current line
        line = lines[num]
        tmp_line = line.rstrip()
        
        if tmp_line.endswith('-'):
        
            tmp_end_spaces = line[len(tmp_line):]
            # next line
            next_line = lines[num + 1].lstrip()
            
            if len(next_line) == 0:
                continue
            
            idx = lines[num + 1].index(next_line[0])
            next_line_spaces = lines[num + 1][:idx]
        
        
            s = next_line.split(maxsplit=1)
            
            if len(s) == 0:
                continue
            elif len(s) == 1:
                lines[num] = tmp_line[:-1] + s[0].strip() + tmp_end_spaces
                lines[num + 1] = next_line_spaces
            elif len(s) >= 2:
                lines[num] = tmp_line[:-1] + s[0].strip() + tmp_end_spaces
                lines[num + 1] = next_line_spaces + s[1]
        
    return ''.join(lines)

def merge_hyphen_across_page(text1, text2):
    
    lines1 = text1.splitlines(keepends=True)
    if len(lines1) == 0:
        return text1, text2
    
    idx1 = len(lines1) - 1
    while idx1 >= 0 and len(lines1[idx1].strip()) == 0:
        idx1 -= 1
    if idx1 < 0:
        return text1, text2
    
    tmp_line = lines1[idx1].rstrip()
    tmp_end_spaces = lines1[idx1][len(tmp_line):]
    
    if not tmp_line.endswith('-'):
        return text1, text2
    
    
    lines2 = text2.splitlines(keepends=True)
    if len(lines2) == 0:
        return text1, text2
    
    idx2 = 0
    while idx2 < len(lines2) and len(lines2[idx2].strip()) == 0:
        idx2 += 1
    if idx2 == len(lines2):
        return text1, text2
    
    next_line = lines2[idx2].lstrip()
    
    
    idx = lines2[idx2].index(next_line[0])
    next_line_spaces = lines2[idx2][:idx]
    
    s = next_line.split(maxsplit=1)
    
    if len(s) == 0:
        return text1, text2
    elif len(s) == 1:
        lines1[idx1] = tmp_line[:-1] + s[0].strip() + tmp_end_spaces
        lines2[idx2] = next_line_spaces
    elif len(s) >= 2:
        lines1[idx1] = tmp_line[:-1] + s[0].strip() + tmp_end_spaces
        lines2[idx2] = next_line_spaces + s[1]
    
    return ''.join(lines1), ''.join(lines2)
    
    

# Include next page
def fix_hyphens(book):
    
    page_bodies = book.findall('.//page_body')
    
    for i in range(len(page_bodies)):
        if page_bodies[i].text is not None:
            page_bodies[i].text = merge_hyphens(page_bodies[i].text)
            
            if i + 1 < len(page_bodies) and page_bodies[i + 1].text is not None:
                page_bodies[i].text, page_bodies[i + 1].text = merge_hyphen_across_page(page_bodies[i].text, page_bodies[i + 1].text)
                
    return book



def clean_text(book):
    for pb in book.findall('.//page_body'):
        if pb.text is not None:
            pb.text = unidecode(pb.text)
            pb.text = pb.text.replace("''", "\"")
            pb.text = pb.text.replace("``", "\"")
    
    return book


def hathitrust_preprocess(input_zip_file, output_dir):
    
    # input_zip_file:  ..../<book_library>/<book_id>.zip
    # Outputs:         output_dir/book_id/raw.xml, output_dir/book_id/base.xml
    
    book_id = os.path.basename(input_zip_file)[:-4]
    lib_id = os.path.basename(os.path.dirname(input_zip_file))
    
    output_dir = Path(output_dir)
    output_dir = output_dir / lib_id / book_id
    unzipped_dir = output_dir / "unzipped/"
    unzipped_loc = unzipped_dir / book_id
    output_path_raw = output_dir / "raw.xml"
    output_path_base = output_dir / "base.xml"
    
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    if not unzipped_dir.exists():
        with zipfile.ZipFile(input_zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzipped_dir)

    if not output_path_raw.exists():
        cleaned_pages = htrc_cleaned_pages(unzipped_loc)
        grouped_sections = parse_grouped_sections(unzipped_loc)
        success = generate_xml_tree(grouped_sections, cleaned_pages, unzipped_loc, output_path_raw)
    
    if not output_path_base.exists():
        parser = etree.XMLParser(remove_blank_text=True)
        book = etree.parse(str(output_path_raw), parser)
        
        book = clean_header_footers(book)
        book = fix_hyphens(book)
        book = clean_text(book)
        
        book.write(str(output_path_base), pretty_print=True, encoding='utf-8')
        
    return