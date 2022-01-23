import lxml.etree as ET
from collections import Counter

def get_lemma_counts(filename):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(filename), parser=parser)
    book = tree.getroot()
    
    tokens = book.findall('.//t')
    lemmas = [t.attrib['lemma'] for t in tokens]
    
    return Counter(lemmas)


def get_non_entity_lemma_counts(filename):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(filename), parser=parser)
    book = tree.getroot()
    
    tokens = book.findall('.//t')
    lemmas = [t.attrib['lemma'] for t in tokens if t.getparent().tag != 'entity']
    
    return Counter(lemmas)


def get_non_character_lemma_counts(filename):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(filename), parser=parser)
    book = tree.getroot()
    
    tokens = book.findall('.//t')
    lemmas = list()
    for t in tokens:
        p = t.getparent()
        if p.tag == 'entity' and 'character' in p.attrib:
            continue
        lemmas.append(t.attrib['lemma'])
    
    
    return Counter(lemmas)


def get_part_of_speech_counts(filename):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(filename), parser=parser)
    book = tree.getroot()
    
    tokens = book.findall('.//t')
    pos = [t.attrib['pos'] for t in tokens]
    
    return Counter(pos)

def get_dependency_tag_counts(filename):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(filename), parser=parser)
    book = tree.getroot()
    
    tokens = book.findall('.//t')
    dep = [t.attrib['dep'] for t in tokens]
    
    return Counter(dep)