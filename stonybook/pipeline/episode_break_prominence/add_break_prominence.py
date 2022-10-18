import os
import lxml.etree as ET
import spacy
from scipy import signal
from collections import defaultdict, Counter
import math
from stonybook.pipeline.episode_break_prominence.chapter_prediction import get_preds

#parse character_coref_annotated.xml and get necessary fields
def parse_xml(input_xml_path, output_dir):
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_xml_path), parser=parser)
    root = tree.getroot()

    lemmas = {}
    para_breaks = {}
    chapter_breaks = {}
    sent_lemma = []
    chapterCounter = 0
    num_tokens = 0

    mainchild = root.find('body')

    #loop over headers (chapters tags) in body
    for header in mainchild:
        for paragraphs in list(header):
            if paragraphs.tag == 'p':
                for sentences in list(paragraphs):
                    if sentences.tag == 's':
                        sent_lemma = {}
                        key = int(sentences.attrib['num'])
                        sent_lemma[key] = []
                        for l in list(sentences):
                            if l.tag == 't':
                                token = int(l.attrib['num'])
                                num_tokens += 1
                                sent_lemma[key].append(l.attrib['lemma'])
                            elif l.tag == 'entity':
                                for entityLemma in list(l):
                                    if entityLemma.tag == 't':
                                        token = int(entityLemma.attrib['num'])
                                        num_tokens += 1
                                        sent_lemma[key].append(entityLemma.attrib['lemma'])

                        lemmas.update(sent_lemma)
                para_breaks[int(paragraphs.attrib['num'])] = key

        chapterCounter += 1
        chapter_breaks[chapterCounter] = [int(paragraphs.attrib['num']), key]

    return para_breaks, lemmas, num_tokens

def get_intersection_length(lemma_dict, x, y):
    x_counts = Counter(lemma_dict[x])
    y_counts = Counter(lemma_dict[y])
    common_lemmas = set(lemma_dict[x]).intersection(set(lemma_dict[y]))
    common_lemmas_count = 0
    for lemma in common_lemmas:
        if lemma.isalnum():
            common_lemmas_count += min(x_counts[lemma], y_counts[lemma])
    return common_lemmas_count

def build_graph(lemma_dict, N = 150):
    edges = defaultdict()
    if len(lemma_dict.keys()) == 0:
        return edges
    for idx in range(max(lemma_dict.keys())):
        for j in range(idx+1, idx+N+1):
            if j not in lemma_dict or idx == j:
                continue
            common_lemmas = get_intersection_length(lemma_dict, idx, j)
            edges[(idx, j)] = common_lemmas
    return edges

#normalize densities in range 1-100
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

#normalize prominences to range 0-1
def normalize_prominences(peaks, prominences):
    normalized_prominence_para = {}
    normalized_prominence = []
    l = len(peaks)

    try:
        maxVal = max(prominences)
        minVal = min(prominences)
    except:
        maxVal = 0
        minVal = 0
    rangeVals = maxVal - minVal
    for i in range(l):
        try:
            val = round((prominences[i] - minVal) / rangeVals, 2)
            normalized_prominence_para[peaks[i]] = val
            normalized_prominence.append(val)
        except ZeroDivisionError:
            continue
    
    return normalized_prominence, normalized_prominence_para

#get densities using inverse exponential weighted function
def get_densities(edges, para_breaks, max_sent_num, N=150):
    density = [0 for _ in range(max_sent_num+1)]
    para_breakSentences = [x for x in para_breaks.values()]
    for i in para_breakSentences:
        for x in range(i-N+1, i+1):
            for y in range(i+1, x+N+1):
                if x < 0 or y > max_sent_num:
                    continue     
                left_dist = i - x + 1
                right_dist = y - i
                density[i] += edges[(x,y)] / math.exp(left_dist + right_dist)
    return normalize_density(density, para_breaks, max_sent_num)

def compute_densities(para_breaks, lemmas):

    stop_words = spacy.load('en_core_web_sm').Defaults.stop_words

    for k in lemmas:
        lemmas[k] = [x for x in lemmas[k] if x not in stop_words]
    
    edges = build_graph(lemmas)

    if len(lemmas.keys()) > 0:
        max_sent_num = max(lemmas.keys())
    else:
        max_sent_num = 0

    densities = get_densities(edges, para_breaks, max_sent_num)

    return densities

#get peaks and prominences
def get_peak_prominences(densities):
    valid_densities = list(densities.values())

    # Get peak indices and prominences
    peaks, _ = signal.find_peaks([-x for x in valid_densities], plateau_size = (0, 5))

    prominences = signal.peak_prominences([-x for x in valid_densities], peaks, wlen=5)[0]
    prominences, normalized_prominences = normalize_prominences(peaks, prominences)
    return peaks, prominences, normalized_prominences

def get_episode_break_prominence(input_xml_path, output_dir):
    para_breaks, lemmas, num_tokens = parse_xml(input_xml_path, output_dir)

    densities = compute_densities(para_breaks, lemmas)

    peaks, prominences, normalized_prominences = get_peak_prominences(densities)

    data = {"densities" : densities, "peaks": peaks, "prominences": prominences, "num_tokens" : num_tokens}

    return data, normalized_prominences

#add chapter break confidences to each paragraph tag
def add_episode_break_prominence(input_xml_path, output_dir, output_xml):
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_xml_path), parser=parser)
    book = tree.getroot()
        
    data, confidences = get_episode_break_prominence(input_xml_path, output_dir)
    
    predictions = get_preds(data)

    paragraph_tags = book.findall('.//p')
    peak_paras = list(confidences.keys())

    for p_tag in paragraph_tags:
        num = int(p_tag.attrib['num'])
        if num in peak_paras:
            p_tag.attrib['episode_break_prominence'] = str(confidences[num])
        else:
            p_tag.attrib['episode_break_prominence'] = "0"

        if num in predictions:
            p_tag.attrib['chapter_break_prediction'] = "1"
        else:
            p_tag.attrib['chapter_break_prediction'] = "0"
    
    #create new xml
    with open(os.path.join(output_dir, output_xml), 'wb') as f:
        f.write(ET.tostring(book, pretty_print=True))