import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from flair.data import Sentence
import spacy
import stanza
import time
from flair.models import MultiTagger
from allennlp.predictors.predictor import Predictor
from spacy.lang.en import English
# spacy.require_gpu()
nlp = English()
nlp.add_pipe("sentencizer")

with open('full_gatsby.txt') as f:
    gatsby_text = f.read()

start = time.time()
doc = nlp(gatsby_text)
end = time.time()
gatsby_sentences = []
for sent in doc.sents:
    gatsby_sentences.append(sent.text)

print(len(gatsby_sentences))
print(end - start, "s")

def test_spacy():
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    start = time.time()
    doc = nlp(gatsby_text)
    end = time.time()
    with open('spacy_ner.txt', 'w') as f:
        for ent in doc.ents:
            f.write("{} {} {} {}".format(ent.text, ent.start_char, ent.end_char, ent.label_) + '\n')
    print("spaCy:", end - start, "s")

def test_stanza():
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, ner', tokenize_no_ssplit=True, use_gpu=True)
    start = time.time()
    doc = stanza_nlp('\n\n'.join(gatsby_sentences))
    end = time.time()
    with open('stanza_ner.txt', 'w') as f:
        for sent in doc.sentences:
            for ent in sent.ents:
                ent_text = ' '.join(ent.text.split())
                f.write("{} {}".format(ent_text, ent.type) + '\n')
    print("Stanza:", end - start, "s")

def test_flair():
    tagger = MultiTagger.load(['pos', 'ner'])
    start = time.time()
    results = []
    N = 64
    sentences = []
    for i, sent in enumerate(gatsby_sentences):
        sentence = Sentence(sent)
        sentences.append(sentence)
        if len(sentences) == N:
            tagger.predict(sentences)
            results.extend([sent.get_spans('ner') for sent in sentences])
    if sentences:
        tagger.predict(sentences)
        results.extend([sent.get_spans('ner') for sent in sentences])
    end = time.time()
    with open('flair_ner.txt', 'w') as f:
        for entities in results:
            for entity in entities:
                f.write(str(entity) + '\n')
    print("Flair:", end - start, "s")

def test_allennlp():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
    start = time.time()
    batch_input = []
    for sent in gatsby_sentences:
        batch_input.append({"sentence": sent})
    results = predictor.predict_batch_json(batch_input)
    end = time.time()
    print("AllenNLP:", end-start, "s")
    with open('allennlp_ner.txt', 'w') as f:
        for sent in results:
            for word, tag in zip(sent["words"], sent["tags"]):
                f.write(f"{word}\t{tag}\n")

# test_spacy()
# test_stanza()
# test_flair()
test_allennlp()