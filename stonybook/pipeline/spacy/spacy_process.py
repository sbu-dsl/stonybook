import spacy
import neuralcoref
import pickle
from lxml import etree
from spacy.tokens import Doc
from operator import itemgetter
from multiprocessing import Pool
from pathlib import Path


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

def gen_chapters_with_tok_info(spacy_results, para_idxs):
    chapters = []
    mentions = []
    global_tok_idx = 0
    for chapter_idx, spacy_result in enumerate(spacy_results):
        chapter_tok_idx = global_tok_idx
        para_points = para_idxs[chapter_idx]
        paragraphs = []
        sentences = []
        para_idx = 0
        for spacy_sentence in spacy_result.sents:
            start_char = spacy_sentence[0].idx
            if para_idx < len(para_points) and start_char >= para_points[para_idx]:
                para_idx += 1
                paragraphs.append(sentences)
                sentences = []
            sentence = []
            for tok in spacy_sentence:
                tok_info = {
                    "text": tok.text, 
                    "lemma": tok.lemma_, 
                    "pos": tok.pos_,
                    "head": str(chapter_tok_idx + tok.head.i),
                    "dep": tok.dep_
                }
                sentence.append(tok_info)
                global_tok_idx += 1
            sentences.append(sentence)
        if sentences:
            paragraphs.append(sentences)
        chapters.append(paragraphs)
        
        for mention in spacy_result.ents:
            tok_start = chapter_tok_idx + mention.start
            tok_end = chapter_tok_idx + mention.end
            mention_info = {
                "tok_start": tok_start,
                "tok_end": tok_end,
                "ner": mention.label_,
                "mention_text": mention.text
            }
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
                    tree_tok.set('dep', token['dep'])
                    tree_tok.set('head', token['head'])

                    if tok_no == mention_start:
                        tree_ner = etree.SubElement(tree_sent, 'entity')
                        tree_ner.set('phrase', local_mention["mention_text"])
                        tree_ner.set('ner', local_mention['ner'])
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

def parse_spacy_coref(spacy_results):
    tok_idxs = []
    for doc in spacy_results:
        mychains = []
        chains = doc._.coref_clusters
        for chain in chains:
            mychain = []
            for mention in chain.mentions:
                idxs = (mention.start, mention.end)
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

def convert_coref_to_tok_idx(coref_tok_idxs, header_tok_starts):
    coref_conv_tok_idxs = []
    for i, coref_chains in enumerate(coref_tok_idxs):
        base_tok_num = header_tok_starts[i]
        chains = []
        for coref_chain in coref_chains:
            tok_idxs = []
            for tstart, tend in coref_chain:
                tok_idxs.append((base_tok_num + tstart, base_tok_num + tend))
            chains.append(tok_idxs)
        coref_conv_tok_idxs.append(chains)
    return coref_conv_tok_idxs

def annotate_coref(annot_xml_path, coref_xml_path, spacy_path):
    with open(spacy_path, "rb") as f:
        spacy_results = pickle.load(f)[0]
    parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = etree.parse(str(annot_xml_path), parser=parser)
    book = tree.getroot()

    header_tok_starts = parse_header_tok_start(book)
    coref_tok_idxs = spacy_results[1]
    coref_conv_tok_idxs = convert_coref_to_tok_idx(coref_tok_idxs, header_tok_starts)
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
    tree.write(str(coref_xml_path), pretty_print=True, encoding='utf-8')

def spacy_single_pickle(book_dir):
    book_dir = Path(book_dir)
    output_path = book_dir / "spacy_annotated.xml"
    output_pkl_path = book_dir / "spacy_annots.pkl"
    input_xml_path = book_dir / "header_annotated.xml"
    if not input_xml_path.exists():
        print(book_dir, " not exists")
        return
    if output_pkl_path.exists():
        print("Cached:", output_pkl_path)
        with open(output_pkl_path, "rb") as f:
            annots, header_attribs, para_idxs = pickle.load(f)
    else:
        nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(nlp)
        sections, header_attribs = parse_headers(input_xml_path)
        chapters, para_idxs = gen_full_chapters_and_para_idxs(sections)
        def annotate_text(nlp, chapters):
            docs = []
            for chapter in chapters:
                docs.append(nlp(chapter))
                break
            coref_data = parse_spacy_coref(docs)
            for doc in docs:
                doc.user_data = {}
            return docs, coref_data
        print("Annotating spacy:", input_xml_path)
        # try:
        annots = annotate_text(nlp, chapters)
        with open(output_pkl_path, "wb") as f:
            pickle.dump((annots, header_attribs, para_idxs), f, pickle.HIGHEST_PROTOCOL)
        print("Finished spacy:", input_xml_path)
    # except Exception as e:
    #     error_path = book_dir / "spacy_error.txt"
    #     with open(error_path, "w") as f:
    #         f.write(str(e))
    #     print("Error:", input_xml_path, str(e))
    #     return
    annot_chapters, chapter_mentions = gen_chapters_with_tok_info(annots[0], para_idxs)
    generate_tokenized_xml(annot_chapters, chapter_mentions, header_attribs, input_xml_path, output_path)
    annotate_coref(output_path, output_path, output_pkl_path)

def spacy_process(book_dirs, num_threads=32):
    with Pool(num_threads) as pool:
        pool.map(spacy_single_pickle, book_dirs)

# def spacy_process(dir):
#     nlp = spacy.load('en_core_web_sm')
#     neuralcoref.add_to_pipe(nlp)

#     doc = nlp(u'Alice Witherspoon has a dog. She loves him.\n\nHowever, she held a much darker secret.')
#     for sent_i, sent in enumerate(doc.sents):
#         print(sent_i)
#         for tok in sent:
#             print(tok.i, tok.text, tok.idx, tok.lemma_, tok.pos_, tok.dep_, tok.head.i)

#     for ent in doc.ents:
#         print(ent.text, ent.start, ent.end, ent.label_)

#     coref_clusters = doc._.coref_clusters
#     for cluster in coref_clusters:
#         for mention in cluster.mentions:
#             start = mention.start
#             end = mention.end
#             print(doc[start:end])
#         print()