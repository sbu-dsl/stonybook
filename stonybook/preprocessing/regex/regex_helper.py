import inflect
from collections import Counter
import re
from itertools import groupby
import pickle
from pathlib import Path


def conv_to_title(string):
    return string[0].upper() + string[1:].lower()

def get_corresponding_rule(rule_text):
    word_numbers = ['word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']
    if rule_text == 'generic_text':
        return ".+"
    elif rule_text == 'desc':
        return "(CHAPTER|Chapter|CHAP|Chap|PART|Part|BOOK|Book|STORY|Story|LETTER|Letter|VOLUME|Volume|VOL|Vol|CASE|Case)([^\S\r\n]*(THE|The|the|NO|No|no|NO\.|No\.|no\.|NUMBER|Number|number|NUMBER\.|Number\.|number\.))*"
    elif rule_text == 'roman_upper':
        return "(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})(?![A-Za-z0-9'\"])"
    elif rule_text == 'roman_lower':
        return "(?=[mdxlxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})(?![A-Za-z0-9'\"])"
    elif rule_text == 'numeral':
        return "[0-9]+"
    elif rule_text == 'punctuation':
        return "[^a-zA-Z\d\s\"']+"
    elif rule_text == 'title_upper':
        return "[^A-Za-z]*[A-Z][^a-z]+"
    elif rule_text == 'title_lower':
        return "[^A-Za-z\d]*[^\S\r\n\D]*[A-Z][^\r\n]*[^\.\?\-!:;,]"
#         return "[^A-Za-z\d]*[\d\s]*[A-Z][\s\S]*[^\.]"
#         return "[\d\s]*[A-Z][\s\S]*[^\.\"']"
#         return "([A-Z\d][^A-Z\s]*[\s]+)*[A-Z\d][^A-Z\s]*"
    elif rule_text == 'whitespace':
        return "[\s]+"
    
    elif rule_text in word_numbers:
        p = inflect.engine()
        if rule_text == 'word_number_one':
            tmp = [p.number_to_words(x) for x in range(201)]
        elif rule_text == 'word_number_One':
            tmp = [p.number_to_words(x).title() for x in range(201)]
            tmp += [conv_to_title(p.number_to_words(x)) for x in range(201)]
        elif rule_text == 'word_number_ONE':
            tmp = [p.number_to_words(x).upper() for x in range(201)]
        elif rule_text == 'word_number_first':
            tmp = [p.ordinal(p.number_to_words(x)) for x in range(201)]
        elif rule_text == 'word_number_First':
            tmp = [p.ordinal(p.number_to_words(x)).title() for x in range(201)]
            tmp += [conv_to_title(p.ordinal(p.number_to_words(x))) for x in range(201)]
        elif rule_text == 'word_number_FIRST':
            tmp = [p.ordinal(p.number_to_words(x)).upper() for x in range(201)]
        
        tmp2 = list()
        for elem in tmp:
            if '-' in elem:
                tmp2.append(elem.replace('-', '[\s]*-[\s]*'))
            else:
                tmp2.append(elem)
        l = sorted(tmp2, key=len)[::-1]
        reg = '|'.join([x + '(?!\-|( and))' for x in l])
        return reg
    
    return None


def generate_sequences(l):
    if len(l) == 0:
        return []
    subsequent = generate_sequences(l[1:])
    answer = list()
    if len(subsequent) > 0:
        answer += subsequent
    for elem in l[0]:
        answer.append([elem])
        for elem2 in subsequent:
            answer.append([elem] + elem2)
    return answer

def remove_duplicates(l):
    res = [] 
    for i in l: 
        if i not in res: 
            res.append(i)
    return res

def remove_consecutives(l):
    res = list()
    for elem in l:
        tmp = [x[0] for x in groupby(elem)]
        if tmp not in res:
            res.append(tmp)
    return res

def remove_whitespace_from_ends(l):
    res = list()
    for elem in l:
        start = 0
        while start < len(elem) and elem[start] == 'whitespace':
            start += 1
        end = len(elem) - 1
        while end >= 0 and elem[end] == 'whitespace':
            end -= 1
        tmp = elem[start:end + 1]
        if tmp and tmp not in res:
            res.append(elem[start:end + 1])
    return res

def issublist(b, a):
    return b in [a[i:len(b)+i] for i in range(len(a))]


def remove_custom_rules(l):
    # desc must be followed by whitespace or punctuation
    res = list()
    for elem in l:
        if 'desc' in elem:
            i = elem.index('desc')
            if i + 1 >= len(elem):
                res.append(elem)
                continue
            if elem[i + 1] not in {'punctuation', 'whitespace'}:
                continue
            res.append(elem)
        else:
            res.append(elem)
    l = res
    
    # blacklisted sequences
    word_numbers = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']
    
    word_numbers_specific = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_first', 'word_number_First']

    res = list()
    for elem in l:
        if issublist(['punctuation', 'whitespace', 'punctuation'], elem):
            continue
        # Redundant
#         if elem == ['punctuation', 'title_lower']:
#             continue
#         if elem == ['punctuation']:
#             continue
        if 'desc' in elem and not any(y in elem for y in word_numbers):
            continue
        # Example: "Three birds"
        if any(y in elem for y in word_numbers_specific):
            if 'desc' not in elem:
                if 'generic_text' in elem:
                    continue
            
#         if elem == ['word_number_One', 'whitespace', 'generic_text']:
#             continue
#         if elem == ['word_number_One', 'generic_text']:
#             continue
#         if elem == ['word_number_First', 'whitespace', 'generic_text']:
#             continue
#         if elem == ['word_number_First', 'generic_text']:
#             continue

        if elem[0] == 'word_number_one':
            continue
        if elem[0] == 'word_number_first':
            continue
        
        # Footnotes
        if elem[0] == 'punctuation' and elem.count('punctuation') == 1:
            continue
        if len(elem) >= 4:
            if elem[0] == 'punctuation' and elem[1] in word_numbers and elem[2] == 'punctuation' and (elem[3:] == ['whitespace', 'generic_text'] or elem[3:] == ['generic_text']):
                continue
        
        res.append(elem)
    l = res
    
    return l

def add_title_generic_to_rules(l):
    res = list()
    res += l
    word_numbers = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']

    for elem in l:
        if any(x in elem for x in word_numbers):
            if 'title_upper' in elem:
                new_rule = [x if x != 'title_upper' else 'generic_text' for x in elem]
                if new_rule not in res:
                    res.append(new_rule)

            if 'title_lower' in elem:
                new_rule = [x if x != 'title_lower' else 'generic_text' for x in elem]
                if new_rule not in res:
                    res.append(new_rule)
                
    return res


def generate_final_regex_rules():
    
#     home = Path.home()
#     stonybook_data_path = home / "stonybook_data"
    
#     if not stonybook_data_path.is_dir():
#         stonybook_data_path.mkdir(parents=True, exist_ok=True)
    
#     regex_cache_file = stonybook_data_path / "regex_rules.pkl"
#     if regex_cache_file.is_file():
#         print('reading pkl')
#         with open(str(regex_cache_file), 'rb') as f:
#             all_seqs, rules, priority = pickle.load(f)
#         print('done reading pkl')
#         return all_seqs, rules, priority
    
    l = list()

    l.append(['desc'])
    l.append(['whitespace'])
    l.append(['punctuation'])
    l.append(['whitespace'])
    l.append(['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST'])
    l.append(['whitespace'])
    l.append(['punctuation'])
    l.append(['whitespace'])
    l.append(['title_upper', 'title_lower'])


    l2 = list()

    l2.append(['word_number_First', 'word_number_FIRST'])
    l2.append(['whitespace'])
    l2.append(['desc'])
    l2.append(['whitespace'])
    l2.append(['punctuation'])
    l2.append(['whitespace'])
    l2.append(['title_upper', 'title_lower'])


    string_to_pattern = dict()

    components = ['desc', 'whitespace', 'punctuation', 'roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST', 'punctuation', 'title_upper', 'title_lower', 'generic_text']

    for elem in components:
        string_to_pattern[elem] = get_corresponding_rule(elem)
        
    
    
    all_seqs = generate_sequences(l) + generate_sequences(l2)
    all_seqs = remove_duplicates(all_seqs)
    all_seqs = remove_consecutives(all_seqs)
    all_seqs = remove_whitespace_from_ends(all_seqs)
    all_seqs = add_title_generic_to_rules(all_seqs)
    all_seqs = remove_custom_rules(all_seqs)
    
    rule_texts = [''.join(['(' + string_to_pattern[x] + ')' for x in y]) for y in all_seqs]
    rules = [re.compile(x) for x in rule_texts]

    priority = ['desc', 'roman_upper', 'roman_lower', 'numeral', 'word_number_first', 'word_number_First', 'word_number_FIRST', 'word_number_one', 'word_number_One', 'word_number_ONE', 'punctuation', 'whitespace', 'title_upper', 'title_lower', 'generic_text']
    
    
#     with open(str(regex_cache_file), 'wb') as f:
#         pickle.dump([all_seqs, rules, priority], f)

    return all_seqs, rules, priority



def get_best_matching_rule(text, all_seqs, rules, priority):
    answers = list()
    for idx in range(len(rules)):
        r = rules[idx]
        if r.fullmatch(text):
            answers.append(all_seqs[idx])
    answers.sort(key=lambda x:[priority.index(y) for y in x])
    return answers



def convert_to_int(number, number_type):
    
    def convert_one_to_int(number):
        p = inflect.engine()
        #number = number.replace(' ', '')
        number = ' '.join(number.strip().split())
        number = '-'.join([x.strip() for x in number.split('-')])
        l = [p.number_to_words(x) for x in range(201)]
        return l.index(number)
    
    def convert_first_to_int(number):
        p = inflect.engine()
        #number = number.replace(' ', '')
        number = ' '.join(number.strip().split())
        number = '-'.join([x.strip() for x in number.split('-')])
        l = [p.ordinal(p.number_to_words(x)) for x in range(201)]
        return l.index(number)
    
    def convert_roman_to_int(number):
        
        def value(r): 
            if (r == 'I'): 
                return 1
            if (r == 'V'): 
                return 5
            if (r == 'X'): 
                return 10
            if (r == 'L'): 
                return 50
            if (r == 'C'): 
                return 100
            if (r == 'D'): 
                return 500
            if (r == 'M'): 
                return 1000
            return -1
        res = 0
        i = 0
        
        number = number.upper()
        while (i < len(number)): 
            # Getting value of symbol s[i]
            s1 = value(number[i])
            if (i+1 < len(number)):
                # Getting value of symbol s[i+1]
                s2 = value(number[i+1])
                # Comparing both values
                if (s1 >= s2):
                    # Value of current symbol is greater
                    # or equal to the next symbol
                    res = res + s1
                    i = i + 1
                else:
                    # Value of current symbol is greater
                    # or equal to the next symbol
                    res = res + s2 - s1
                    i = i + 2
            else:
                res = res + s1
                i = i + 1
        return res
    
    if number_type == 'numeral':
        return int(number)
    if number_type.lower() == 'word_number_one':
        return convert_one_to_int(number.lower())
    if number_type.lower() == 'word_number_first':
        return convert_first_to_int(number.lower())
    if number_type.startswith('roman'):
        return convert_roman_to_int(number)
    
    return 0


def get_header_attrs(text, rule):
    
    desc = None
    number = None
    number_text = None
    number_type = None
    title = None
    rule_text = ','.join(rule)
    
    if rule_text == 'title_upper':
        return desc, number, number_text, number_type, text, rule_text
    
    number_list = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST', 'numeral']
    
    curr_text = text
    
    for element in rule:
        
        if element == 'whitespace':
            curr_text = curr_text.lstrip()
            continue
        
        r = re.compile(get_corresponding_rule(element))
        
        m = r.match(curr_text)
        
        start, end = m.span()
        
        if element == 'desc':
            desc = curr_text[start:end].strip()
        
        elif element in number_list:
            number = convert_to_int(curr_text[start:end].strip(), element)
            number_text = curr_text[start:end].strip()
            number_type = element
        
        elif element in ['title_upper', 'title_lower']:
            title = curr_text[start:end].strip()
        
        curr_text = curr_text[end:]
    
    return desc, number, number_text, number_type, title, rule_text
        
        






