import stanza
from stanza.server import CoreNLPClient
import time

with open('full_gatsby.txt') as f:
    gatsby_text = f.read()[:100000]

with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner'],
        timeout=300000,
        memory='6G') as client:
    start = time.time()
    ann = client.annotate(gatsby_text)
    end = time.time()
    start = time.time()
    ann = client.annotate(gatsby_text)
    end = time.time()
    print("CoreNLP:", end - start, "s")

stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, ner', use_gpu=True)
start = time.time()
doc = stanza_nlp(gatsby_text)
end = time.time()
print("Stanza:", end - start, "s")
print(len(doc.sentences))

