import lxml.etree as ET
from stonybook.preprocessing.hathitrust.header import convert_base_to_header_annot


def add_nesting(book):
    # Assumes that there are only 'header' and 'p' tags in 'body'
    
    headers = book.findall('.//header')
    if len(headers) == 0:
        body = book.find('.//body')
        elem = ET.Element('header')
        elem.set('desc', str(None))
        elem.set('number', str(None))
        elem.set('number_text', str(None))
        elem.set('number_type', str(None))
        elem.set('title', str(elem.text))
        elem.set('rule_text', 'init_header')
        body.insert(0, elem)
    
    headers = book.findall('.//header')
    
    for h in headers:
        h.set('text', str(h.text))
        h.text = None
        
        curr_elem = h.getnext()
        while curr_elem is not None and curr_elem.tag == 'p':
            curr_elem_2 = curr_elem.getnext()
            h.append(curr_elem)
            curr_elem = curr_elem_2
    
    
    return book


def add_dummy_header(book):
    body = book.find('.//body')
    
    children = body.getchildren()
    if len(children) > 0:
        if children[0].tag == 'header':
            return book
    else:
        return book
    elem = ET.Element('header')
    elem.set('desc', str(None))
    elem.set('number', str(None))
    elem.set('number_text', str(None))
    elem.set('number_type', str(None))
    elem.set('title', str(elem.text))
    elem.set('rule_text', 'init_header')
    body.insert(0, elem)
    
    return book


def annotate_headers(book, regex_tuple):
    
    # Get rid of page header and page footer tags
    ET.strip_elements(book, 'page_header')
    ET.strip_elements(book, 'page_footer')
    # Get rid of page body tags but keep the text
    ET.strip_tags(book, 'page_body')
    # Get rid of page tags but keep the text
    pages = book.findall('.//page')
    for p in pages:
        if p.text is not None:
            p.text += '\n'
    ET.strip_tags(book, 'page')
    
    
    # Make each line its own paragraph
    body = book.find('.//body')
    
    text = body.text
    lines = text.splitlines(keepends=True)
    
    body.text = ''
    
    for l in lines:
        ET.SubElement(body, "p").text = l
    
    
    # Run header annot code
    book = convert_base_to_header_annot(book, regex_tuple)
    
    
    # Merge consecutive paras if eligible
    body = book.find('.//body')
    children = body.getchildren()
    to_merge = list()
    for i in range(len(children) - 1):
        curr_child = children[i]
        next_child = children[i + 1]
        
        if curr_child.tag == 'header' or next_child.tag == 'header':
            continue
        
        t = curr_child.text
        tn = next_child.text
        
        if len(t) >= 2 and len(tn) >= 1:
            if t[-1] == '\n' and t[-2] in {'.', '?', '!', '"', "'"}:
                if tn[0] in {'"', "'"} or tn[0].isupper():
                    continue
        
        to_merge.append(i)
    
    to_merge = sorted(to_merge, reverse=True)
    
    # Merge paragraphs
    for para_num in to_merge:
        next_para_num = para_num + 1
        children[para_num].text += children[next_para_num].text
        children[next_para_num].getparent().remove(children[next_para_num])
    
    body = book.find('.//body')
    paras = body.findall('.//p')
    for p in paras:
        p.text = p.text.strip()
    
    book = add_nesting(book)
    book = add_dummy_header(book)
    

    return book