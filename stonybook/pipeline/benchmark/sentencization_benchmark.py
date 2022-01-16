import time
import spacy
import stanza
from flair.data import Sentence
from flair.models import MultiTagger

spacy_nlp = spacy.load("en_core_web_sm")

with open('full_gatsby.txt') as f:
    gatsby_text = f.read()

def test_spacy():
    start = time.time()
    doc = spacy_nlp(gatsby_text)
    end = time.time()
    result = [sentence.text for sentence in doc.sents]
    with open('spacy_sents.txt', 'w') as f:
        for sent in result:
            sent = ' '.join(sent.split())
            if not sent:
                continue
            f.write(sent + '\n')
    print("spaCy:", end - start, "s")

def test_stanza():
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize')
    start = time.time()
    doc = stanza_nlp(gatsby_text)
    end = time.time()
    result = [sentence.text for sentence in doc.sentences]
    with open('stanza_sents.txt', 'w') as f:
        for sent in result:
            sent = ' '.join(sent.split())
            if not sent:
                continue
            f.write(sent + '\n')
    print("Stanza:", end - start, "s")

test_spacy()
test_stanza()