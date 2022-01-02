import coreferee, spacy
gpu = spacy.require_gpu()
print('GPU:', gpu)

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
doc = nlp("Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.")

print(doc._.coref_chains.print())
# doc[16]._.coref_chains.print()
# doc._.coref_chains.resolve(doc[31])

# gpu = spacy.prefer_gpu()
# print('GPU:', gpu)

# nlp = spacy.load("en_core_web_sm")

# doc = nlp("This is a sentence. This is another sentence.")
# for sent in doc.sents:
#     print(sent.text)

# import spacy
# import neuralcoref
# import time
# from stanza.server import CoreNLPClient

# with open('full_gatsby.txt') as f:
#     gatsby_text = f.read()[:100]

# # with CoreNLPClient(
# #         annotators=['tokenize','ssplit','pos','lemma','ner', 'depparse', 'coref'],
# #         timeout=300000,
# #         memory='8G') as client:
# #     start = time.time()
# #     ann = client.annotate(gatsby_text)
# #     end = time.time()
# #     start = time.time()
# #     ann = client.annotate(gatsby_text)
# #     end = time.time()
# #     print("CoreNLP:", end - start, "s")

# spacy.require_gpu()

# nlp = spacy.load('en')
# neuralcoref.add_to_pipe(nlp)

# start = time.time()
# doc = nlp(gatsby_text)
# end = time.time()
# print("SpaCy:", end - start, "s")