import xml.etree.ElementTree as ET
import re
import pandas as pd
from unidecode import unidecode
import spacy
import pickle
from spacy.matcher import Matcher

def identifyMatchingGroups(x):
    space_replace_exceptions = ["ll", "d", "am", "ve", "m", "s", "t", "re"]

    n_groups = len(x.groups())

    if n_groups > 0:
        if x.group(1) == "'":
            if n_groups >= 2:
                # For (')(one or two characters)

                if x.group(2) not in space_replace_exceptions:
                    return x.group(1) + " " + x.group(2)
                else:
                    return "`" + x.group(2)
            else:
                return x.group(1)
        elif x.group(2) == "'":
            if n_groups >= 4:
                # For (atleast one non space character)(')(space)(word)
                if x.group(4) not in space_replace_exceptions:
                    return x.group(1) + " " + x.group(2) + " " + x.group(4)
                else:
                    return x.group(1) + '`' + x.group(4)

        else:
            # other than single quote
            return x.group(1) + " " + x.group(2)
    else:
        # No groups found for any regex match
        return x


def normalizeQuotes(text_tokens):
    """ normalize the quotes """

    new_tokens = []

    for token in text_tokens:
        transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-`", u"'''\"\"--'")])
        unidecode_output = unidecode(token.text, errors='strict')
        token.text = unidecode_output.translate(transl_table)
        new_tokens.append(token)

    return new_tokens


def addSpacesToQuotes(text):
    """ adds space before and after the quotes """

    # adding space before and after quotes
    regex_quotes = [r"(\S)(“)", r"(\S)(”)", r'(\S)(")', r"(\S)(“)", r"(\S)(‘)", r"(\S)(’)", r"(\S{1})(')(\s*)(\w{1,2})",
                    r"(“)(\S)", r"(”)(\S)", r'(")(\S)', r"(“)(\S)", r"(‘)(\S)", r"(’)(\S)", r"(')(\w{1,2})"]

    for idx, quote in enumerate(regex_quotes):
        match = re.search(quote, text)
        if match is not None:
            text = re.sub(quote, lambda x: identifyMatchingGroups(x), text)

    return text

reportingVerbs = open('reportingVerbs.txt','r').read().split("\n")


class TrigramMatching():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = [
            # Quote Verb Mention
            [
                {'POS': 'PUNCT'},
                {'POS': 'VERB'},
                {'POS': 'NOUN'}
            ],
            [
                {'POS': 'PUNCT'},
                {'POS': 'VERB'},
                {'POS': 'PROPN'}
            ],
            # Quote Mention Verb
            [
                {'POS': 'PUNCT'},
                {'POS': 'NOUN'},
                {'POS': 'VERB'}

            ],
            [
                {'POS': 'PUNCT'},
                {'POS': 'PRON'},
                {'POS': 'VERB'}
            ],
            # Mention Verb Quote
            [
                {'POS': 'NOUN'},
                {'POS': 'VERB'},
                {'POS': 'PUNCT'}

            ],
            [
                {'POS': 'PRON'},
                {'POS': 'VERB'},
                {'POS': 'PUNCT'}
            ],
            # Verb Mention Quote
            [
                {'POS': 'VERB'},
                {'POS': 'NOUN'},
                {'POS': 'PUNCT'}
            ],
            [
                {'POS': 'VERB'},
                {'POS': 'PRON'},
                {'POS': 'PUNCT'}
            ],

        ]

    def getMentionFromPattern(self, pattern, tokens):
        noun = None
        tokens = tokens.split()
        for i, kvPair in enumerate(pattern):
            if kvPair['POS'] == 'NOUN' or kvPair['POS'] == 'PRON':
                noun = tokens[i]
                break
        return noun

    def getMention(self, text):
        for i, pattern in enumerate(self.patterns):
            matcher = Matcher(self.nlp.vocab)
            matcher.add(f"Trigram_{i}", [pattern])

            doc = self.nlp(text)

            matches = matcher(doc)
            for match_id, start, end in matches:
                string_id = self.nlp.vocab.strings[match_id]
                span = doc[start:end]
                return self.getMentionFromPattern(pattern, span.text)

patterns = [
               [
                   {'ORTH': ','},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': '!'}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': '!'}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': '?'}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': '?'}
               ],
                   [
                   {'ORTH': ','},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': '.'}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': '.'}
               ],
                   [
                   {'ORTH': ','},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': ';'}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': ';'}
               ],
                   [
                   {'ORTH': ','},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': ','}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': ','}
               ],
                   [
                   {'ORTH': '"'},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': ','}
               ],
               [
                   {'ORTH': '"'},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': ','}
               ],
                   [
                   {'ORTH': ','},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': '"'}
               ],
               [
                   {'ORTH': ','},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': '"'}
               ],
                   [
                   {'ORTH': 'dear'},
                   {'POS': 'PROPN', 'OP': '+'}
               ],
               [
                   {'ORTH': 'dear'},
                   {'POS': 'NOUN', 'OP': '+'}
               ],
                   [
                   {'ORTH': 'Dear'},
                   {'POS': 'PROPN', 'OP': '+'}
               ],
               [
                   {'ORTH': 'Dear'},
                   {'POS': 'NOUN', 'OP': '+'}
               ],
                   [
                   {'ORTH': 'Oh'},
                   {'POS': 'PROPN', 'OP': '+'},
                   {'ORTH': '!'}
               ],
               [
                   {'ORTH': 'Oh'},
                   {'POS': 'NOUN', 'OP': '+'},
                   {'ORTH': '!'}
               ],
]

from spacy.tokens import Span


class POSMatcher():
    def __init__(self, patterns, doc=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = patterns
        self.doc = doc

    def getMentionFromPattern(self, pattern, tokens):
        noun = None
        tokens = tokens.split()
        for i, kvPair in enumerate(pattern):
            if kvPair['POS'] == 'NOUN' or kvPair['POS'] == 'PRON':
                noun = tokens[i]
                break
        return noun

    def getSpacyTokens(self, startId, endId):
        return self.doc[startId:endId+1]

    def getValueAtIndex(self, span, alignment, index):
        return " ".join([span[idx].text for idx, i in enumerate(alignment) if i == 1])

    def getMatches(self, quoteObj):

        # To store the result of last match
        result = None
        # To store the results of matches found
        outputs = []

        for i, pattern in enumerate(self.patterns):
            matcher = Matcher(self.nlp.vocab)
            matcher.add(f"{i}", [pattern])

            # doc = self.getSpacyTokens(quoteObj.quote_start, quoteObj.quote_end)
            doc = self.nlp(quoteObj.quote)

            matches = matcher(doc, with_alignments=True)

            # No match found, continue to check other patterns
            if len(matches) == 0:
                continue

            # When a match is found, check for the speakers
            for match_id, start, end, alignment in matches[-1:]:

                span = None
                if end + 1 > len(doc):
                    span = Span(doc, start, end, label=match_id)
                else:
                    end = end + 1
                    span = Span(doc, start, end, label=match_id)

                # Holds the speaker in the current match
                mention_start = -1
                mention_end = -1
                #                 n =
                for i in range(len(span)):
                    if i < len(alignment):
                        if alignment[i] == 1 and mention_start == -1:
                            mention_start = start + i
                        if alignment[i] == 1:
                            mention_end = start + i

                current_speaker = self.getValueAtIndex(span, alignment, 1)
                mention_start = quoteObj.quote_start
                mention_end = quoteObj.quote_start

                # Check if there any matches found already
                if len(outputs) > 0:

                    ### Three cases
                    ### if current speaker is a subset of all prev speakers, store the new result
                    ### else if any of the prev speaker is a subset of current speaker, store the prev speaker result
                    ### else if current speaker does not match with prev speakers, return None

                    def check_speaker_in_found_match(speaker, outputs):
                        for output in outputs:
                            # is current speaker a subset of the prev speaker
                            if speaker not in output['speaker']:
                                return False
                        return True

                    def check_found_match_in_speaker(speaker, outputs):
                        for output in outputs:
                            # is prev speaker a subset of current speaker
                            if output['speaker'] in speaker:
                                return True, output
                        return False, []

                    if check_speaker_in_found_match(current_speaker, outputs):
                        # current speaker is a subset of previous results, store the current match result
                        result = {'span': span,
                                  'speaker': self.getValueAtIndex(span, alignment, 1),
                                  'pattern': pattern[1]['POS'],
                                  'mention_start': mention_start,
                                  'mention_end': mention_end}
                        outputs = [result]
                        continue
                    else:
                        flag, result = check_found_match_in_speaker(current_speaker, outputs)
                        if flag:
                            # current speaker is super set of a previous result, store the found previous result
                            outputs = [result]
                            continue
                        else:
                            # No match between current and prev speaker, return none (Handle ambiguity)
                            return None

                            # found the first match, store the result
                result = {'span': span,
                          'speaker': self.getValueAtIndex(span, alignment, 1),
                          'pattern': pattern[1]['POS'],
                          'mention_start': mention_start,
                          'mention_end': mention_end}
                outputs.append(result)

        if len(outputs) == 0:
            return None
        return outputs[-1]

class Token:
    def __init__(self, text, tokenId, paragraphId, sentenceId, pos, lemma, dep, children, head,
                 corefStart=None, corefEnd=None, ner=None):
        self.text = text
        self.tokenId = tokenId
        self.paragraphId = paragraphId
        self.sentenceId = sentenceId
        self.pos = pos
        self.lemma = lemma
        self.ner = ner
        self.dep = dep
        self.head = head
        self.children = children
        self.corefStart = corefStart
        self.corefEnd = corefEnd

    def __str__(self):
        return f"Token(text : {self.text}, tokenId : {self.tokenId}, sentId : {self.sentenceId}, paraId : {self.paragraphId}, )"


def getTokenChildren(token_list):
    """ returns all the children for a given token """

    parent_to_children_dict = {}
    for token in token_list:
        parent = int(token.head)
        if parent in parent_to_children_dict:
            parent_to_children_dict[parent].append(token)
        else:
            parent_to_children_dict[parent] = [token]
    return parent_to_children_dict


def getTokenList(character_coref_file_path):
    """ returns a list of token objects within the xml file"""

    root = ET.parse(character_coref_file_path).getroot()

    tokens_list = []

    for book in root.iter('book'):
        for body in book.iter('body'):
            for header in body.iter('header'):
                for p in header.iter('p'):
                    for s in p.iter('s'):
                        for t in s.iter('t'):
                            children = []
                            coref_start, coref_end = None, None
                            if 'coref_tok_num_start' in t.attrib:
                                coref_start = t.attrib['coref_tok_num_start']
                            if 'coref_tok_num_end' in t.attrib:
                                coref_end = t.attrib['coref_tok_num_end']
                            token = Token(text=t.text,
                                          paragraphId=int(p.attrib['num']),
                                          sentenceId=int(s.attrib['num']),
                                          tokenId=int(t.attrib['num']),
                                          lemma=t.attrib.get('lemma', None),
                                          pos=t.attrib.get('pos', None),
                                          ner=t.attrib.get('ner', None),
                                          dep=t.attrib.get('dep', None),
                                          head=t.attrib.get('head', None),
                                          children=children,
                                          corefStart=coref_start,
                                          corefEnd=coref_end
                                          )
                            tokens_list.append(token)

    parent_to_child_dict = getTokenChildren(tokens_list)

    new_tokens_list = []

    for i, token in enumerate(tokens_list):
        if token.tokenId in parent_to_child_dict:
            token.children = parent_to_child_dict[token.tokenId]
        else:
            token.children = []
        new_tokens_list.append(token)

    return new_tokens_list


class Quote:
    def __init__(self, quote_start=0,
                 quote_end=0,
                 mention_start=-1,
                 mention_end=-1,
                 mention=None,
                 char_id=-1,
                 quote='',
                 startSentId=0,
                 endSentId=0,
                 start_token=None,
                 end_token=None,
                 paragraphId=-1):
        self.quote_start = quote_start
        self.quote_end = quote_end
        self.mention_start = mention_start
        self.mention_end = mention_end
        self.mention = mention
        self.char_id = char_id
        self.quote = quote
        self.startSentId = startSentId
        self.endSentId = endSentId
        self.start_token = start_token
        self.end_token = end_token
        self.paragraphId = paragraphId
        self.quoteListIdx = -1

    def __str__(self):
        return f"Mention : {self.mention} \n \
                 Quote : {self.quote} \n \
                 Quote Start : {self.start_token}\n \
                 Quote End : {self.end_token}\n \
                 startSentId : {self.startSentId}\n \
                 endSentId : {self.endSentId}\n \
                 paragraphId : {self.paragraphId}\n \
                 mention_start : {self.mention_start} \n \
                 mention_end : {self.mention_end}\n"


class StonyBookQuote:

    def __init__(self, tokens_list, patterns, book_dir):
        self.tokens = tokens_list
        self.tokenDict = {}
        self.conversations = None
        self.prevDelimiters = {}
        # self.doc = self.loadSpacyDoc(book_dir)
        # self.posMatcher = POSMatcher(patterns, self.doc)
        # self.posMatcher = POSMatcher(patterns)
        self.trigramMatcher = TrigramMatching()
        self.tokenDict = {t.tokenId: t for t in self.tokens}
        self.quotes = None
        self.reportingVerbs = reportingVerbs
        self.extractQuotes()
        self.quoteParaDict = {}
        self.generateQuoteParaDict()

    def normalizeTokens(self, token):
        return token

    def loadSpacyDoc(self, book_dir):
        with open(str(book_dir) + '/spacy_annots.pkl', "rb") as f:
            annots, header_attribs, para_idxs = pickle.load(f)
            return annots[0][0]

    def getTokens(self):
        return self.tokens

    def writeQuotesToFile(self, filename):
        quotes = []
        for quote in self.quotes:
            quotes.append(quote.__dict__)
        df = pd.DataFrame(quotes, columns=['quote_start', 'quote_end', 'mention_start', 'mention_end', 'mention_phrase',
                                           'char_id', 'quote', 'startSentId', 'endSentId'])
        df.to_csv(filename, sep='\t', index=False)

    def writeEntitiesToFile(self, filename):
        entities_dict = [{'COREF': -1, 'start_token': -1, 'end_token': -1, 'prop': "$", 'cat': "$", 'text': "$"}]
        entities_df = pd.DataFrame(entities_dict)
        entities_df.to_csv(filename, index=False, sep="\t")

    def tokensBeforeInSameSentence(self, tokenId, sentenceId):
        i = tokenId
        prefix = []
        while i >= 0 and self.tokenDict[i].sentenceId == sentenceId:
            prefix = [self.tokenDict[i]] + prefix
            i = i - 1
        return prefix

    def tokensAfterInSameSentence(self, tokenId, sentenceId):
        i = tokenId
        suffix = []
        while i in self.tokenDict and self.tokenDict[i].sentenceId == sentenceId:
            suffix += [self.tokenDict[i]]
            i = i + 1
        return suffix

    def dependencyParsing(self):
        delimiters = set(['.', ';', ':', "'", '"', ')'])
        for i, quote in enumerate(self.quotes):
            quoteTokens = []
            if quote.start_token.tokenId - 1 in self.tokenDict:
                prevToken = self.tokenDict[quote.start_token.tokenId - 1]
                # Check if previous token is in same paragraph or not.
                if prevToken.paragraphId == quote.start_token.paragraphId:
                    if prevToken.text not in delimiters and prevToken.sentenceId == quote.start_token.sentenceId:
                        prefixTokens = self.tokensBeforeInSameSentence(quote.start_token.tokenId - 1,
                                                                       quote.start_token.sentenceId)
                        quoteTokens = prefixTokens

            # Next Token
            quoteTokens += [self.tokenDict[x] for x in range(quote.quote_start, quote.quote_end + 1)]
            if quote.end_token.tokenId + 1 in self.tokenDict:
                nextToken = self.tokenDict[quote.end_token.tokenId + 1]
                prevToken = self.tokenDict[quote.end_token.tokenId - 1]
                if nextToken.paragraphId == quote.end_token.paragraphId:
                    if prevToken.text != "." and nextToken.sentenceId == quote.end_token.sentenceId:
                        quoteTokens += self.tokensAfterInSameSentence(quote.end_token.tokenId + 1,
                                                                      quote.end_token.sentenceId)
                    elif prevToken.text != "." and nextToken.sentenceId > quote.end_token.sentenceId:
                        quoteTokens += self.tokensAfterInSameSentence(quote.end_token.tokenId + 1,
                                                                      quote.end_token.sentenceId + 1)

            candidates = []

            def outsideQuote(token):
                if token.tokenId < quote.quote_start or token.tokenId > quote.quote_end:
                    return True
                return False

            for token in quoteTokens:
                if outsideQuote(token):
                    if (token.pos.startswith("VB") and token.lemma.lower() in self.reportingVerbs):
                        candidates.append(token)

            def getTokenDistance(token):
                if token.tokenId < quote.quote_start:
                    return abs(token.tokenId - quote.quote_start)
                return abs(token.tokenId - quote.quote_end)

            mention = None

            if len(candidates) == 0:
                self.quotes[i].mention = None
                continue

            candidates.sort(key=lambda x: getTokenDistance(x))
            reportingVerb = candidates[0]

            def expand_noun(tok):
                compounds = [
                    child
                    for tc in [tok]
                    for child in tc.children
                    if child.dep == "compound"
                ]
                return [tok] + compounds

            children = [child for child in reportingVerb.children]
            filtered_children = [x for x in children if
                                 x.tokenId < quote.start_token.tokenId or x.tokenId > quote.end_token.tokenId]
            childrenPOS = [child.pos for child in filtered_children]
            mention_start = mention_end = -1
            if 'NNP' in childrenPOS:
                cIdx = childrenPOS.index('NNP')
                expanded_list = expand_noun(filtered_children[cIdx])
                tmp = [t.text for t in expanded_list]
                tmp.reverse()
                mention = " ".join(tmp)
                mention_start = expanded_list[-1].tokenId
                mention_end = expanded_list[0].tokenId
            elif 'NNPS' in childrenPOS:
                cIdx = childrenPOS.index('NNPS')
                expanded_list = expand_noun(filtered_children[cIdx])
                tmp = [t.text for t in expanded_list]
                tmp.reverse()
                mention = " ".join(tmp)
                mention_start = expanded_list[-1].tokenId
                mention_end = expanded_list[0].tokenId
            elif 'NN' in childrenPOS:
                cIdx = childrenPOS.index('NN')
                expanded_list = expand_noun(filtered_children[cIdx])
                tmp = [t.text for t in expanded_list]
                tmp.reverse()
                mention = " ".join(tmp)
                mention_start = expanded_list[-1].tokenId
                mention_end = expanded_list[0].tokenId
            elif 'NNS' in childrenPOS:
                cIdx = childrenPOS.index('NNS')
                expanded_list = expand_noun(filtered_children[cIdx])
                tmp = [t.text for t in expanded_list]
                tmp.reverse()
                mention = " ".join(tmp)
                mention_start = expanded_list[-1].tokenId
                mention_end = expanded_list[0].tokenId
            elif 'PRP' in childrenPOS:
                cIdx = childrenPOS.index('PRP')
                expanded_list = expand_noun(filtered_children[cIdx])
                tmp = [t.text for t in expanded_list]
                tmp.reverse()
                mention = " ".join(tmp)
                mention_start = expanded_list[-1].tokenId
                mention_end = expanded_list[0].tokenId
            self.quotes[i].mention = mention
            self.quotes[i].mention_start = mention_start
            self.quotes[i].mention_end = mention_end

    def getSentence(self, sentenceId):
        sentenceTokens = []
        for token in self.tokens:
            if token.sentenceId == sentenceId:
                sentenceTokens.append(token)
        return sentenceTokens

    def getTrigrams(self):
        cols = ['quote_start', 'quote_end', 'quote']
        for quote in self.quotes:
            quoteStart = quote.quote_start
            quoteEnd = quote.quote_end
            startSentId = self.tokenDict[quoteStart].sentenceId
            endSentId = self.tokenDict[quoteEnd].sentenceId

    def vocativeDetection(self):
        # Detect quotes with Vocatives and the next quote.
        vocativeQuotes = []
        for i, quoteObj in enumerate(self.quotes):

            match = self.posMatcher.getMatches(quoteObj)
            if match is not None:
                if i + 1 < len(self.quotes):
                    nextQuote = self.quotes[i + 1]
                    sentDiff = int(nextQuote.startSentId) - int(quoteObj.endSentId)
                    isSameSpeaker = False
                    if int(sentDiff) < 3 and \
                            isSameSpeaker == False and \
                            self.quotes[i+1].mention is None:
                        if self.quotes[i+1].mention is None:
                            self.quotes[i+1].mention = match['speaker']
                            self.quotes[i+1].mention_start = match['mention_start']
                            self.quotes[i+1].mention_end = match['mention_end']

    def findMostUsedQuote(self):
        """ finds the most used quote in the given tokens """

        quotes_counter = {'double': 0, 'single': 0}

        tokens = self.getTokens()

        for token in tokens:
            if token.text == '"':
                quotes_counter['double'] += 1
            elif token.text == "'":
                quotes_counter['single'] += 1

        most_used_quote = max(quotes_counter, key=quotes_counter.get)

        if most_used_quote == 'double':
            return '"'
        else:
            return "'"

    def getQuotes(self):
        return self.quotes

    def extractQuotes(self):
        """ identifies quotes from the given list of tokens """

        most_used_quote = self.findMostUsedQuote()
        tokens = self.getTokens()
        last_para_id = None
        quotes = []
        quote_object = Quote()

        quote_start_token = None
        current_quote = []

        for token in tokens:

            quote_start_flag = False
            for char in token.text:
                if char == most_used_quote:
                    quote_start_flag = True

            if last_para_id is not None and token.paragraphId != last_para_id:
                if len(current_quote) > 0:
                    quote_object.quote_start = quote_start_token
                    quote_object.quote_end = token.tokenId - 1
                    quote_object.quote = ' '.join(current_quote)
                    quote_object.startSentId = self.tokenDict[quote_object.quote_start].sentenceId
                    quote_object.endSentId = self.tokenDict[quote_object.quote_end].sentenceId
                    quote_object.start_token = self.tokenDict[quote_object.quote_start]
                    quote_object.end_token = self.tokenDict[quote_object.quote_end]
                    quote_object.paragraphId = quote_object.start_token.paragraphId
                    quotes.append(quote_object)

                    current_quote.append(token.text)

                quote_start_token = None
                current_quote = []
                quote_object = Quote()

            if quote_start_flag:
                if quote_start_token is not None:
                    if len(current_quote) > 0:
                        current_quote.append(token.text)
                        quote_object.quote_start = quote_start_token
                        quote_object.quote_end = token.tokenId
                        quote_object.quote = ' '.join(current_quote)
                        quote_object.startSentId = self.tokenDict[quote_object.quote_start].sentenceId
                        quote_object.endSentId = self.tokenDict[quote_object.quote_end].sentenceId
                        quote_object.start_token = self.tokenDict[quote_object.quote_start]
                        quote_object.end_token = self.tokenDict[quote_object.quote_end]
                        quote_object.paragraphId = quote_object.start_token.paragraphId
                        quotes.append(quote_object)

                    current_quote = []
                    quote_start_token = None
                    quote_object = Quote()
                else:
                    quote_start_token = token.tokenId

            if quote_start_token is not None:
                current_quote.append(token.text)

            last_para_id = token.paragraphId

        self.quotes = quotes

    def generateQuoteParaDict(self):
        for quoteObj in self.quotes:
            if quoteObj.paragraphId in self.quoteParaDict:
                self.quoteParaDict[quoteObj.paragraphId].append(quoteObj)
            else:
                self.quoteParaDict[quoteObj.paragraphId] = [quoteObj]

    def continuousQuotesWithSameSpeaker(self):
        for i, quoteObj in enumerate(self.quotes):
            currentQuote = quoteObj
            nextQuote = None
            if i + 1 < len(self.quotes):
                nextQuote = self.quotes[i + 1]
            if nextQuote is not None and currentQuote.end_token.paragraphId == nextQuote.start_token.paragraphId:
                tokenStart = currentQuote.quote_end
                tokenEnd = nextQuote.quote_start

                textTokens = [v for k, v in self.tokenDict.items() if v.tokenId >= tokenStart and v.tokenId <= tokenEnd]
                flagReportingVerb = False
                for token in textTokens:
                    if token.pos.startswith("VB") and token.lemma.lower() in self.reportingVerbs:
                        flagReportingVerb = True
                        break
                if flagReportingVerb == False:
                    continue

                mention = currentQuote.mention
                mention_start = currentQuote.mention_start
                mention_end = currentQuote.mention_end

                # Current <- Next
                if mention is None:
                    if nextQuote.mention is not None:
                        self.quotes[i].mention = nextQuote.mention
                        self.quotes[i].mention_start = nextQuote.mention_start
                        self.quotes[i].mention_end = nextQuote.mention_end
                        mention = nextQuote.mention
                        mention_start = nextQuote.mention_start
                        mention_end = nextQuote.mention_end
                    continue
                candidateQuoteIds = []
                for qObj in self.quoteParaDict[currentQuote.paragraphId]:
                    if qObj.quoteListIdx > currentQuote.quoteListIdx:
                        candidateQuoteIds.append(qObj.quoteListIdx)

                for qIdx in candidateQuoteIds:
                    # Current -> Next
                    if mention is not None:
                        if self.quotes[qIdx].mention is None:
                            self.quotes[qIdx].mention = mention
                            self.quotes[qIdx].mention_start = mention_start
                            self.quotes[qIdx].mention_end = mention_end
                            continue

    def createConversationCsv(self, basepath, filename):
        conversationQuotes = []
        for convo in self.conversations:
            quotesMerged = ""
            for each in convo:
                quotesMerged += f"{each.mention} : " + each.quote + "\n"
            conversationQuotes.append({'quotes': quotesMerged})
        df = pd.DataFrame(conversationQuotes)
        df.to_csv(basepath + filename + '.csv')

    def getConversations(self):
        conversations = []
        currentConversation = []

        for i, quoteObj in enumerate(self.quotes):
            if len(currentConversation) == 0:
                currentConversation.append(quoteObj)
            else:
                previousSentenceId = int(currentConversation[-1].endSentId)
                if int(quoteObj.startSentId) - previousSentenceId <= 3:
                    currentConversation.append(quoteObj)
                else:
                    conversations.append(currentConversation)
                    currentConversation = [quoteObj]

        self.conversations = conversations

    def searchUpwards(self, conversationIdx, currentIdx):
        i = currentIdx - 2
        while (i >= 0):
            if self.conversations[conversationIdx][i].mention != None:
                self.conversations[conversationIdx][currentIdx].mention = self.conversations[conversationIdx][i].mention
                self.conversations[conversationIdx][currentIdx].mention_start = self.conversations[conversationIdx][
                    i].mention_start
                self.conversations[conversationIdx][currentIdx].mention_end = self.conversations[conversationIdx][
                    i].mention_end
                return True
            i = i - 2
        return False

    def searchDownwards(self, conversationIdx, currentIdx):
        i = currentIdx + 2
        while (i < len(self.conversations[conversationIdx])):
            if self.conversations[conversationIdx][i].mention != None:
                self.conversations[conversationIdx][currentIdx].mention = self.conversations[conversationIdx][i].mention
                self.conversations[conversationIdx][currentIdx].mention_start = self.conversations[conversationIdx][
                    i].mention_start
                self.conversations[conversationIdx][currentIdx].mention_end = self.conversations[conversationIdx][
                    i].mention_end
                return True
            i = i + 2
        return False

    def correctMentionsInConversations(self):
        for i, conversation in enumerate(self.conversations):
            for j, quoteObj in enumerate(conversation):
                if quoteObj.mention == None:
                    status = self.searchUpwards(i, j)
                    if status == False:
                        self.searchDownwards(i, j)
        self.updateQuotes()

    def updateQuotes(self):
        for conversation in self.conversations:
            for quoteObj in conversation:
                self.quotes[quoteObj.quoteListIdx].mention = quoteObj.mention
                self.quotes[quoteObj.quoteListIdx].mention_start = quoteObj.mention_start
                self.quotes[quoteObj.quoteListIdx].mention_end = quoteObj.mention_end

    def populateQuoteListIdx(self):
        for i, quoteObj in enumerate(self.quotes):
            self.quotes[i].quoteListIdx = i

    def looseConversation(self):
        for quoteObj in self.quotes:
            if quoteObj.mention is None:
                mention = None
                # Search Upwards
                idx = quoteObj.quoteListIdx - 1
                while (idx >= 0):
                    if self.quotes[idx].mention is not None:
                        self.quotes[quoteObj.quoteListIdx].mention = self.quotes[idx].mention
                        self.quotes[quoteObj.quoteListIdx].mention_start = self.quotes[idx].mention_start
                        self.quotes[quoteObj.quoteListIdx].mention_end = self.quotes[idx].mention_end
                        break
                    idx = idx - 1

    def dumpXML(self, input_xml_path, output_xml_path):
        # Writes deterministic sieves output to character coref xml file

        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        # Temporary quote (for syntax)
        temp_quote = root.find('./analysis/quotes/quote')

        # To remove the existing quotes information
        for analysis_1 in root.iter('analysis'):
            for quotes_1 in analysis_1.iter('quotes'):
                analysis_1.remove(quotes_1)

        # Added quotes empty tag
        for analysis in root.iter('analysis'):
            ET.SubElement(analysis, "quotes").text = "\n"

        # Added quote tags
        for analysis_2 in root.iter('analysis'):
            for quotes_2 in analysis_2.iter('quotes'):
                for quote_2 in self.quotes:
                    ET.SubElement(quotes_2, "quote",
                                  mention=quote_2.mention,
                                  tokBegin=str(quote_2.quote_start),
                                  tokEnd=str(quote_2.quote_end),
                                  mentionBegin=str(quote_2.mention_start),
                                  mentionEnd=str(quote_2.mention_end))

        tree.write(output_xml_path, encoding='utf-8' ,pretty_print=True)

def run(book_dir, input_xml_path, output_xml_path):
    tokens_list = getTokenList(input_xml_path)
    tokens_list = normalizeQuotes(tokens_list)
    obj = StonyBookQuote(tokens_list, patterns, book_dir)
    obj.dependencyParsing()
    obj.populateQuoteListIdx()
    obj.continuousQuotesWithSameSpeaker()
    # Unable to use this sieve due to Spacy Versioning Issues
    # obj.vocativeDetection()
    obj.getConversations()
    obj.correctMentionsInConversations()
    obj.looseConversation()
    obj.dumpXML(input_xml_path, output_xml_path)
