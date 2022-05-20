import os
import lxml.etree as ET
import pickle
from nltk.corpus import stopwords
from scipy import signal
import math

def parse_xml(input_xml_path, output_dir):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_xml_path), parser=parser)
    root = tree.getroot()

    lemmas = {}
    para_breaks = {}
    chapter_breaks = {}
    sent_lemma = []

    for mainchild in root:
        chapterCounter = 0
        if mainchild.tag == 'body':
            for var in mainchild:
                cur = var.attrib

                for paragraphs in list(var):
                    if paragraphs.tag == 'p':
                        for sentences in list(paragraphs):
                            if sentences.tag == 's':
                                sent_lemma = {}
                                key = int(sentences.attrib['num'])
                                sent_lemma[key] = []
                                for l in list(sentences):
                                    if l.tag == 't':
                                        token = int(l.attrib['num'])
                                        sent_lemma[key].append(l.attrib['lemma'])
                                    elif l.tag == 'entity':
                                        for entityLemma in list(l):
                                            if entityLemma.tag == 't':
                                                token = int(entityLemma.attrib['num'])
                                                sent_lemma[key].append(entityLemma.attrib['lemma'])

                                lemmas.update(sent_lemma)
                        para_breaks[int(paragraphs.attrib['num'])] = key

                if 'desc' in cur.keys():
                    chapterCounter += 1
                    chapter_breaks[chapterCounter] = [int(paragraphs.attrib['num']), key]

    with open(os.path.join(output_dir, 'lemmas_pickle.pkl'), 'wb') as f:
        pickle.dump(lemmas, f) 

    return para_breaks, chapter_breaks

def build_graph(lemma_dict, para_breaks, N = 150):
    edges = list()
    para_breakSentences = [x for x in para_breaks.values()]
    if len(lemma_dict.keys()) == 0:
        return edges
    for idx in range(max(lemma_dict.keys()) + 1):
        if idx not in para_breakSentences:
            continue
        for n in range(-N, 1 + N):
            if idx + n not in lemma_dict or idx == n:
                continue

            common_lemmas = lemma_dict[idx].intersection(lemma_dict[idx + n])
            new_common_lemmas = set()
            for x in common_lemmas:
                if x not in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~":
                    new_common_lemmas.add(x)
            common_lemmas = new_common_lemmas
            for i in range(len(common_lemmas)):
                edges.append([idx, idx + n])
    return edges

def normalize_density(densities, para_breaks, max_sent_num):
    para_breakSentences = [x for x in para_breaks.values()]
    numParas = len(para_breakSentences)
    densities = [densities[x] for x in range(max_sent_num+1) if x in para_breakSentences]

    density = {x:0 for x in range(0, numParas)}
    try:
        maxDensity = max(densities)
        minDensity = min(densities)
    except ValueError:
        maxDensity = 0
        minDensity = 0
    rangeofDensities = maxDensity - minDensity
    for i in range(numParas):
        try:
            density[i] = ((densities[i] - minDensity)*100) / rangeofDensities
        except ZeroDivisionError:
            density[i] = 100
    return dict(density)

def get_densities(edges, para_breaks, max_sent_num):
    density = [0 for _ in range(max_sent_num+1)]
    for x, y in edges:
        for i in range(x, y):
            left_dist = i - x + 1
            right_dist = y - i
            density[i] += 1 / math.exp(left_dist + right_dist)
    return normalize_density(density, para_breaks, max_sent_num)

def compute_densities(input_xml_path, output_dir, para_breaks, chapter_breaks):
    with open(os.path.join(output_dir, 'lemmas_pickle.pkl'), 'rb') as f:
        lemmas = pickle.load(f)

    stop_words = set(stopwords.words('english'))

    for k in lemmas:
        lemmas[k] = set(lemmas[k])
        lemmas[k] = lemmas[k].difference(stop_words)
    
    edges = build_graph(lemmas, para_breaks)

    try:
        max_sent_num = max(lemmas.keys())
    except ValueError:
        max_sent_num = 0

    densities = get_densities(edges, para_breaks, max_sent_num)

    os.remove(os.path.join(output_dir, 'lemmas_pickle.pkl'))
    return densities

def get_peak_prominences(densities):
    valid_densities = list(densities.values())

    # Get peak indices and prominences
    peaks, _ = signal.find_peaks([-x for x in valid_densities], plateau_size = (0, 5))

    prominences = signal.peak_prominences([-x for x in valid_densities], peaks, wlen=5)[0]

    return dict(zip(peaks, prominences))


def get_episode_break_prominence(input_xml_path, output_dir):
    para_breaks, chapter_breaks = parse_xml(input_xml_path, output_dir)

    densities = compute_densities(input_xml_path, output_dir, para_breaks, chapter_breaks)
    # print(densities)

    prominences = get_peak_prominences(densities)

    return prominences

#add chapter break confidences to each paragraph tag
def add_episode_break_prominence(input_xml_path, output_dir, output_xml):
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_xml_path), parser=parser)
    book = tree.getroot()
        
    confidences = get_episode_break_prominence(input_xml_path, output_dir)

    paragraph_tags = book.findall('.//p')
    peak_paras = list(confidences.keys())

    for p_tag in paragraph_tags:
        num = int(p_tag.attrib['num'])
        if num in peak_paras:
            p_tag.attrib['episode_break_prominence'] = str(confidences[num])
        else:
            p_tag.attrib['episode_break_prominence'] = "0"
    
    #create new xml
    with open(os.path.join(output_dir, output_xml), 'wb') as f:
        f.write(ET.tostring(book, pretty_print=True))