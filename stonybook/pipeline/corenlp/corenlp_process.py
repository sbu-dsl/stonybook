from lxml import etree
import pickle
from pathlib import Path
from operator import itemgetter
from multiprocessing import Pool
from stanza.server import CoreNLPClient, StartServer
from unidecode import unidecode
import stanza.protobuf.CoreNLP_pb2 as pb2
import re

def gen_full_chapters_and_para_idxs(sections):
    chapters = []
    para_char_end_idxs = []
    for section in sections:
        chapters.append('\n\n'.join(section))
        idx = 0
        idxs = []
        for para in section:
            idx += len(para) + 2 # 2 for newlines
            idxs.append(idx)
        para_char_end_idxs.append(idxs)
    return chapters, para_char_end_idxs


def parse_headers(header_xml_path):
    root = etree.parse(str(header_xml_path))
    body = root.find('body')
    sections = []
    header_attribs = []
    for header in body.iter('header'):
        paragraphs = []
        for para in header.iter('p'):
            paragraphs.append(para.text)
        if len(paragraphs) <= 1:
            continue
        sections.append(paragraphs)
        header_attrib = dict(header.attrib)
        header_attribs.append(header_attrib)
    return sections, header_attribs

def gen_chapters_with_tok_info(corenlp_results, para_idxs):
    chapters = []
    tok_ner = []
    mentions = []
    global_tok_idx = 0
    for chapter_idx, corenlp_result in enumerate(corenlp_results):
        chapter_tok_idx = global_tok_idx
        para_points = para_idxs[chapter_idx]
        paragraphs = []
        sentences = []
        para_idx = 0
        for corenlp_sentence in corenlp_result.sentence:
            start_char = corenlp_sentence.token[0].beginChar
            if para_idx < len(para_points) and start_char >= para_points[para_idx]:
                para_idx += 1
                paragraphs.append(sentences)
                sentences = []
            sentence = []
            
            # Adding dependency parse attributes
            # Subtracting 1 because corenlp indexing starts from 1
            sentence_start_tok_index = global_tok_idx

            parent = dict()
            for r in corenlp_sentence.basicDependencies.root:
                parent[r - 1] = (-1, 'null')
            for e in corenlp_sentence.basicDependencies.edge:
                parent[e.target - 1] = (e.source - 1 + sentence_start_tok_index, e.dep)
            
            for tok_idx, tok in enumerate(corenlp_sentence.token):
                tok_info = {
                    "text": tok.word, 
                    "lemma": tok.lemma, 
                    "pos": tok.pos, 
                    "ner": tok.ner,
                    "head": str(parent[tok_idx][0]) if tok_idx in parent else '-1',
                    "dep": parent[tok_idx][1] if tok_idx in parent else 'null'
                }
                tok_ner.append(tok.ner)
                sentence.append(tok_info)
                global_tok_idx += 1

            sentences.append(sentence)
        if sentences:
            paragraphs.append(sentences)
        chapters.append(paragraphs)
        
        for mention in corenlp_result.mentions:
            tok_start = chapter_tok_idx + mention.tokenStartInSentenceInclusive
            tok_end = chapter_tok_idx + mention.tokenEndInSentenceExclusive
            if tok_ner[tok_start] == "O":
                continue
            mention_info = {
                "tok_start": tok_start,
                "tok_end": tok_end,
                "ner": mention.ner,
                "mention_text": mention.entityMentionText
            }
            if mention.HasField('timex'):
                timex = mention.timex
                mention_info['timex'] = (timex.value, timex.text, timex.type)
            mentions.append(mention_info)

    mentions = sorted(mentions, key=itemgetter('tok_start'))
    return chapters, mentions

def generate_tokenized_xml(chapters, mentions, chapter_tags, base_xml_path, xml_path):
    with open(base_xml_path, 'r') as f:
        base_root = etree.fromstring(f.read())
    root = etree.Element("book")
    base_meta = base_root.find("meta")
    if base_meta != None:
        meta = etree.SubElement(root, "meta")
        for child in base_root.find("meta"):
            meta.append(child)
    analysis = etree.SubElement(root, "analysis")
    body = etree.SubElement(root, "body")
    tok_no = 0
    sent_no = 0
    para_no = 0
    section_no = 0
    mention_idx = 0
    if mentions:
        local_mention = mentions[mention_idx]
        mention_start = local_mention["tok_start"]
        mention_end = local_mention["tok_end"]
    for i, chapter in enumerate(chapters):
        tree_chap = etree.SubElement(body, 'header')
        for key in chapter_tags[i]:
            tree_chap.set(key, chapter_tags[i][key])
        section_no += 1
        for paragraph in chapter:
            tree_para = etree.SubElement(tree_chap, 'p')
            tree_para.set('num', str(para_no))
            para_no += 1
            for sentence in paragraph:
                tree_sent = etree.SubElement(tree_para, 's')
                tree_sent.set('num', str(sent_no))
                sent_no += 1
                tree_ner = None
                for token in sentence:
                    tree_tok = etree.Element('t')
                    tree_tok.set('num', str(tok_no))

                    tree_tok.text = token['text']
                    tree_tok.set('lemma', token['lemma'])
                    tree_tok.set('pos', token['pos'])
                    tree_tok.set('ner', token['ner'])
                    tree_tok.set('dep', token['dep'])
                    tree_tok.set('head', token['head'])

                    if tok_no == mention_start:
                        tree_ner = etree.SubElement(tree_sent, 'entity')
                        tree_ner.set('phrase', local_mention["mention_text"])
                        tree_ner.set('ner', local_mention['ner'])
                        if 'timex' in local_mention:
                            tval, ttype, ttext = local_mention['timex']
                            tree_ner.set('timex_value', tval) 
                            tree_ner.set('timex_type', ttype)
                            tree_ner.set('timex_text', ttext)
                    if mention_start <= tok_no < mention_end:
                        tree_ner.append(tree_tok)
                    else:
                        tree_sent.append(tree_tok)
                    if tok_no == mention_end - 1:
                        tree_ner = None
                        mention_idx += 1
                        prev_end = mention_end
                        while mention_idx < len(mentions):
                            local_mention = mentions[mention_idx]
                            mention_start = local_mention["tok_start"]
                            mention_end = local_mention["tok_end"]
                            if mention_start >= prev_end:
                                break
                            mention_idx += 1
                        
                    tok_no += 1

    book = etree.ElementTree(root)
    tree_output = etree.tostring(book, encoding='utf-8', pretty_print=True)
    with open(xml_path, "wb") as f:
        f.write(tree_output)
    return tree_output

def parse_corenlp_quotes(corenlp_results):
    quote_infos = []
    for result in corenlp_results:
        quote_info = []
        for quote in result.quote:
            quote_info.append(
                (quote.text,
                 quote.mention,
                 quote.tokenBegin, 
                 quote.tokenEnd, 
                 quote.mentionBegin,
                 quote.mentionEnd)
            )
        quote_infos.append(quote_info)
    return quote_infos

def parse_corenlp_coref(corenlp_results):
    tok_idxs = []
    for ann in corenlp_results:
        mychains = []
        chains = ann.corefChain
        for chain in chains:
            mychain = []
            for mention in chain.mention:
                idxs = (mention.sentenceIndex, mention.beginIndex, mention.endIndex)
                mychain.append(idxs)
            mychain.sort()
            mychains.append(mychain)
        tok_idxs.append(mychains)
    return tok_idxs

def parse_header_tok_start(book):
    tok_starts = []
    for header in book.find('body'):
        for tok in header.iter('t'):
            tok_starts.append(int(tok.get('num')))
            break
    return tok_starts

def parse_header_sent_start(book):
    sent_starts = []
    for header in book.find('body'):
        for sent in header.iter('s'):
            sent_starts.append(int(sent.get('num')))
            break
    return sent_starts

def map_sent_to_tok_num(book):
    sent_to_tok = {}
    for sent in book.iter('s'):
        sent_num = sent.get('num')
        for tok in sent.iter('t'):
            tok_num = tok.get('num')
            sent_to_tok[int(sent_num)] = int(tok_num)
            break
    return sent_to_tok

def convert_coref_to_tok_idx(coref_tok_idxs, header_sent_starts, sent_to_tok_num):
    coref_conv_tok_idxs = []
    for i, coref_chains in enumerate(coref_tok_idxs):
        base_sent_num = header_sent_starts[i]
        chains = []
        for coref_chain in coref_chains:
            tok_idxs = []
            for sent_idx, tstart, tend in coref_chain:
                base_tok_num = sent_to_tok_num[base_sent_num + sent_idx]
                tok_idxs.append((base_tok_num + tstart, base_tok_num + tend))
            chains.append(tok_idxs)
        coref_conv_tok_idxs.append(chains)
    return coref_conv_tok_idxs


def annotate_quotes(annot_xml_path, quote_xml_path, corenlp_path):
    with open(corenlp_path, "rb") as f:
        corenlp_results = [pb2.Document.FromString(s) for s in pickle.load(f)[0]]
    section_quotes = parse_corenlp_quotes(corenlp_results)
    parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = etree.parse(str(annot_xml_path), parser=parser)
    book = tree.getroot()
    header_tok_starts = parse_header_tok_start(book)
    analysis = book.find('analysis')
    quotes_tree = etree.SubElement(analysis, 'quotes')
    for i, quotes in enumerate(section_quotes):
        start_tok_num = int(header_tok_starts[i])
        for text, mention, tokbegin, tokend, mentionbegin, mentionend in quotes:
            quote = etree.SubElement(quotes_tree, 'quote')
            quote.set('mention', mention)
            quote.set('tokBegin', str(start_tok_num + tokbegin))
            quote.set('tokEnd', str(start_tok_num + tokend))
            quote.set('mentionBegin', str(start_tok_num + mentionbegin))
            quote.set('mentionEnd', str(start_tok_num + mentionend))

    header_sent_starts = parse_header_sent_start(book)
    coref_tok_idxs = parse_corenlp_coref(corenlp_results)
    sent_to_tok_num = map_sent_to_tok_num(book)
    coref_conv_tok_idxs = convert_coref_to_tok_idx(coref_tok_idxs, header_sent_starts, sent_to_tok_num)
    tok_num_to_parent = {}
    for coref_chains in coref_conv_tok_idxs:
        for coref_chain in coref_chains:
            init_tstart, init_tend = coref_chain[0]
            for tstart, tend in coref_chain[1:]:
                for i in range(tstart, tend):
                    tok_num_to_parent[i] = ((init_tstart, init_tend))
    for tok in book.iter('t'):
        tok_num = int(tok.get('num'))
        if tok_num in tok_num_to_parent:
            init_tstart, init_tend = tok_num_to_parent[tok_num]
            tok.set('coref_tok_num_start', str(init_tstart))
            tok.set('coref_tok_num_end', str(init_tend))
    tree.write(str(quote_xml_path), pretty_print=True, encoding='utf-8')

def corenlp_single_pickle(book_dir):
    output_path = book_dir / "corenlp_annotated.xml"
    output_pkl_path = book_dir / "corenlp_annots.pkl"
    input_xml_path = book_dir / "header_annotated_4.xml"
    if not input_xml_path.exists():
        input_xml_path = book_dir / "header_annotated_3.xml"
    if not input_xml_path.exists():
        input_xml_path = book_dir / "header_annotated_2.xml"
    if not input_xml_path.exists():
        input_xml_path = book_dir / "header_annotated.xml"
    if not input_xml_path.exists():
        print(book_dir, " not exists")
        return
    if output_pkl_path.exists():
        print("Cached:", output_pkl_path)
        with open(output_pkl_path, "rb") as f:
            annots, header_attribs, para_idxs = pickle.load(f)
    else:
        sections, header_attribs = parse_headers(input_xml_path)
        chapters, para_idxs = gen_full_chapters_and_para_idxs(sections)
        
        port_num=8895
        with CoreNLPClient(
            start_server=StartServer.DONT_START,
            endpoint="http://localhost:{}".format(port_num)
        ) as client:
            def annotate_text(chapters):
                annots = []
                for i, chapter in enumerate(chapters):
                    ann = client.annotate(chapter)
                    annots.append(ann.SerializeToString())
                return annots
            print("Annotating CoreNLP:", input_xml_path)
            try:
                annots = annotate_text(chapters)
                with open(output_pkl_path, "wb") as f:
                    pickle.dump((annots, header_attribs, para_idxs), f, pickle.HIGHEST_PROTOCOL)
                print("Finished CoreNLP:", input_xml_path)
            except Exception as e:
                error_path = book_dir / "corenlp_error.txt"
                with open(error_path, "w") as f:
                    f.write(str(e))
                print("Error:", input_xml_path, str(e))
                return
    corenlp_results = [pb2.Document.FromString(s) for s in annots]
    annot_chapters, chapter_mentions = gen_chapters_with_tok_info(corenlp_results, para_idxs)
    generate_tokenized_xml(annot_chapters, chapter_mentions, header_attribs, input_xml_path, output_path)
    annotate_quotes(output_path, output_path, output_pkl_path)


def corenlp_pickle(book_dirs, num_threads=32):
    # pickle corenlp annotations for all dirs in book_dirs
    max_char_length = 10000000
    port_num=8895
    with CoreNLPClient(
        annotators=["tokenize", "ssplit", "pos", "lemma", "ner", "depparse", "coref", "quote"], 
        timeout=6000000,
        max_char_length=max_char_length,
        threads=num_threads,
        memory="128G",
        be_quiet=True,
        endpoint="http://localhost:{}".format(port_num)
    ) as client:
        with Pool(num_threads) as pool:
            pool.map(corenlp_single_pickle, book_dirs)