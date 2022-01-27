import lxml.etree as ET
import pandas as pd
from pathlib import Path
import json

def generate_tokens_csv(directory_name):
    directory_name = Path(directory_name)
    
    input_filename = directory_name / "character_coref_annotated.xml"
    
    if not input_filename.exists():
        print('Please provide a directory name in which "character_coref_annotated.xml" exists.')
        return
    
    output_csv = directory_name / "tokens.csv"
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_filename), parser=parser)
    book = tree.getroot()
    
    l = list()
    
    
    paras = book.findall('.//p')
    for p in paras:
        para_num = int(p.attrib['num'])
        sents = p.findall('.//s')
        first_tok_num_in_para = int(sents[0].find('.//t').attrib['num'])
        for s in sents:
            sent_num = int(s.attrib['num'])
            tokens = s.findall('.//t')
            first_tok_num_in_sent = int(tokens[0].attrib['num'])
            for t in tokens:
                token_num = int(t.attrib['num'])
                
                row = list()
                row.append(para_num)
                row.append(sent_num)
                row.append(token_num)
                row.append(token_num - first_tok_num_in_para)
                row.append(token_num - first_tok_num_in_sent)
                row.append(t.text)
                row.append(str(t.attrib['lemma']))
                row.append(str(t.attrib['pos']))
                row.append(str(t.attrib['dep']))
                row.append(int(t.attrib['head']))
                row.append(str(t.attrib['ner']))
                
                l.append(row)
    
    headings = ['para_num', 'sent_num', 'tok_num_in_doc', 'tok_num_in_para', 'tok_num_in_sent', 'word', 'lemma', 'POS_tag', 'dependency_relation', 'dependency_head_token_num', 'NER']
    
    df = pd.DataFrame(l, columns=headings)
    
    df.to_csv(str(output_csv), index=False)
    
    
def generate_entity_csv(directory_name):
    
    directory_name = Path(directory_name)
    
    input_filename = directory_name / "character_coref_annotated.xml"
    
    if not input_filename.exists():
        print('Please provide a directory name in which "character_coref_annotated.xml" exists.')
        return
    
    output_csv = directory_name / "entities.csv"
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_filename), parser=parser)
    book = tree.getroot()
    
    l = list()
    
    entities = book.findall('.//entity')
    
    # phrase, token start, token end, ner
    
    for e in entities:
        toks = e.findall('.//t')
        row = list()
        row.append(int(toks[0].attrib['num']))
        row.append(int(toks[-1].attrib['num']))
        row.append(str(e.attrib['phrase']))
        row.append(str(e.attrib['ner']))
        l.append(row)
    
    headings = ['start_tok_num', 'end_tok_num', 'phrase', 'ner']
    
    df = pd.DataFrame(l, columns=headings)
    
    df.to_csv(str(output_csv), index=False)

    
def generate_character_json(directory_name):
    
    directory_name = Path(directory_name)
    
    input_filename = directory_name / "character_coref_annotated.xml"
    
    if not input_filename.exists():
        print('Please provide a directory name in which "character_coref_annotated.xml" exists.')
        return
    
    output_json = directory_name / "characters.json"
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_filename), parser=parser)
    book = tree.getroot()
    
    d = dict()
    
    characters = book.find('.//characters').findall('character')
    
    for c in characters:
        
        char_id = int(c.attrib['id'])
        d[char_id] = dict()
        
        
        if 'first_name' in c.attrib and 'last_name' in c.attrib:
            d[char_id]['first_name'] = str(c.attrib['first_name'])
            d[char_id]['last_name'] = str(c.attrib['last_name'])
        else:
            d[char_id]['generic_name'] = str(c.attrib['generic_name'])
        
        if 'gender' in c.attrib:
            d[char_id]['gender'] = c.attrib['gender']
            
        d[char_id]['count'] = int(c.attrib['count'])
        d[char_id]['gendered_coref_count'] = int(c.attrib['gendered_coref_count'])
        d[char_id]['first_person_coref_count'] = int(c.attrib['first_person_coref_count'])
        d[char_id]['second_person_coref_count'] = int(c.attrib['second_person_coref_count'])
        
        d[char_id]['names'] = dict()
        
        names = c.findall('name')
        
        for n in names:
            d[char_id]['names'][n.text] = {'gendered_coref_count': int(n.attrib['gendered_coref_count']), \
                                          'first_person_coref_count': int(n.attrib['gendered_coref_count']), \
                                          'second_person_coref_count': int(n.attrib['second_person_coref_count'])}
        
    with open(str(output_json), "w") as outfile:
        json.dump(d, outfile)
            
        
# <quote mention="Oliver" tokBegin="564" tokEnd="574" mentionBegin="512" mentionEnd="512"/>
def generate_quotes_csv(directory_name):
    
    directory_name = Path(directory_name)
    
    input_filename = directory_name / "character_coref_annotated.xml"
    
    if not input_filename.exists():
        print('Please provide a directory name in which "character_coref_annotated.xml" exists.')
        return
    
    output_csv = directory_name / "quotes.csv"
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_filename), parser=parser)
    book = tree.getroot()
    
    l = list()
    
    quotes = book.find('.//quotes').findall('quote')
    
    for q in quotes:
        row = list()
        row.append(str(q.attrib['mention']))
        row.append(int(q.attrib['tokBegin']))
        row.append(int(q.attrib['tokEnd']))
        row.append(int(q.attrib['mentionBegin']))
        row.append(int(q.attrib['mentionEnd']))
        
        l.append(row)
    
    
    headings = ['mention', 'begin_tok_num', 'end_tok_num', 'begin_mention_tok_num', 'end_mention_tok_num']
    
    df = pd.DataFrame(l, columns=headings)
    
    df.to_csv(str(output_csv), index=False)