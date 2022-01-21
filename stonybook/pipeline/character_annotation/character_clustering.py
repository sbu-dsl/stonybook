import lxml.etree as ET
import pandas as pd
from collections import Counter, defaultdict
from gender_guesser.detector import Detector

def subfinder(mylist, pattern):
    indices = list()
    l = len(pattern)
    for idx in range(len(mylist) - l):
        if mylist[idx:idx+l] == pattern:
            indices.append(idx)
    return indices, l

def merge_consecutive(l):
    prev_num = -1
    curr_text = ''
    curr_toks = list()
    ans = list()
    for num, text in l:
        if prev_num == num - 1:
            curr_text += ' ' + text
            curr_toks.append(num)
        else:
            if len(curr_toks) > 0:
                ans.append([curr_toks, curr_text.strip()])
            curr_text = text
            curr_toks = [num]
        prev_num = num
    
    if len(curr_toks) > 0:
        ans.append([curr_toks, curr_text.strip()])
    return ans

def parse_gender(string):
    if 'female' in string:
        return 'female'
    if 'male' in string:
        return 'male'
    return None

def get_gender_from_honorific(x, male_honorifics, female_honorifics):
    if x in male_honorifics:
        return 'male'
    if x in female_honorifics:
        return 'female'
    return None

def parse_name(string, honorifics):
    d = defaultdict(lambda: None)
    names = string.split()
    if names[0] in honorifics:
        d['honorific'] = names[0]
        names = names[1:]
    if len(names) == 1:
        d['generic_name'] = names[0]
        return d
    
    if len(names) == 2:
        if len(names[0]) == 2 and names[0][-1] == '.':
            d['first_initial'] = names[0][0]
        else:
            d['first_name'] = names[0]
        
        if len(names[1]) == 2 and names[1][-1] == '.':
            d['last_initial'] = names[1][0]
        else:
            d['last_name'] = names[1]
        
        return d
    
    if len(names) == 3:
        if len(names[0]) == 1 or (len(names[0]) == 2 and names[0][-1]) == '.':
            d['first_initial'] = names[0][0]
        else:
            d['first_name'] = names[0]
        
        if len(names[1]) == 1 or (len(names[1]) == 2 and names[1][-1] == '.'):
            d['middle_initial'] = names[1][0]
        else:
            d['middle_name'] = names[1]
        
        if len(names[2]) == 1 or (len(names[2]) == 2 and names[2][-1] == '.'):
            d['last_initial'] = names[2][0]
        else:
            d['last_name'] = names[2]
        
        return d
    
    d['generic_name'] = string
    return d


def get_gender_from_coref(name, male_counts, female_counts):
    if male_counts[name] > 1.1 * female_counts[name]:
        return 'male'
    elif female_counts[name] > 1.1 * male_counts[name]:
        return 'female'
    return None



def find_index_if_present(idx, use_g, l, must_match=None):
    
    if use_g is None:
        if must_match is not None:
            # Previous
            for index, g_hon, g_name, g_coref in l[::-1]:
                if index < idx and g_hon == must_match:
                    return index
            # Next
            for index, g_hon, g_name, g_coref in l:
                if index > idx and g_hon == must_match:
                    return index
            # Previous
            for index, g_hon, g_name, g_coref in l[::-1]:
                if index < idx and g_coref == must_match:
                    return index
            # Next
            for index, g_hon, g_name, g_coref in l:
                if index > idx and g_coref == must_match:
                    return index
            # Previous
            for index, g_hon, g_name, g_coref in l[::-1]:
                if index < idx and g_name == must_match:
                    return index
            # Next
            for index, g_hon, g_name, g_coref in l:
                if index > idx and g_name == must_match:
                    return index
            # Remaining
            # Previous
            for index, g_hon, g_name, g_coref in l[::-1]:
                if index < idx:
                    ret_bool = True
                    if g_hon is not None and g_hon != must_match:
                        ret_bool = False
                    if g_coref is not None and g_coref != must_match:
                        ret_bool = False
                    if g_name is not None and g_name != must_match:
                        ret_bool = False
                    if ret_bool:
                        return index
            # Next
            for index, g_hon, g_name, g_coref in l:
                if index > idx:
                    ret_bool = True
                    if g_hon is not None and g_hon != must_match:
                        ret_bool = False
                    if g_coref is not None and g_coref != must_match:
                        ret_bool = False
                    if g_name is not None and g_name != must_match:
                        ret_bool = False
                    if ret_bool:
                        return index
        else:
            # Previous
            for index, g_hon, g_name, g_coref in l[::-1]:
                if index < idx:
                    return index
            # Next
            for index, g_hon, g_name, g_coref in l:
                if index > idx:
                    return index
        return None
    
    # Honorific to honorific
    # Previous
    for index, g_hon, g_name, g_coref in l[::-1]:
        if index < idx and g_hon is not None and use_g == g_hon:
            return index
    # Next
    for index, g_hon, g_name, g_coref in l:
        if index > idx and g_hon is not None and use_g == g_hon:
            return index

    # Honorific to coref
    # Previous
    for index, g_hon, g_name, g_coref in l[::-1]:
        if index < idx and g_coref is not None and use_g == g_coref:
            if g_hon is None or use_g == g_hon:
                return index
    # Next
    for index, g_hon, g_name, g_coref in l:
        if index > idx and g_coref is not None and use_g == g_coref:
            if g_hon is None or use_g == g_hon:
                return index

    # Honorific to name
    # Previous
    for index, g_hon, g_name, g_coref in l[::-1]:
        if index < idx and g_name is not None and use_g == g_name:
            if g_hon is None or use_g == g_hon:
                if g_coref is None or use_g == g_coref:
                    return index
    # Next
    for index, g_hon, g_name, g_coref in l:
        if index > idx and g_name is not None and use_g == g_name:
            if g_hon is None or use_g == g_hon:
                if g_coref is None or use_g == g_coref:
                    return index
    
    return None


def find_matching_index(idx, row, l):
    
    if row['gender_from_honorific'] is not None:
        # Honorific to honorific, coref, name
        use_g = row['gender_from_honorific']
        index = find_index_if_present(idx, use_g, l)
        if index is not None:
            return index
    
    if row['gender_from_coref'] is not None:
        # Coref to honorific, coref, name
        use_g = row['gender_from_coref']
        index = find_index_if_present(idx, use_g, l)
        if index is not None:
            return index
    
    if row['gender_from_name'] is not None:
        # Name to honorific, coref, name
        use_g = row['gender_from_name']
        index = find_index_if_present(idx, use_g, l)
        if index is not None:
            return index
    
    g = None
    if row['gender_from_name'] is not None:
        g = row['gender_from_name']
    if row['gender_from_coref'] is not None:
        g = row['gender_from_coref']
    if row['gender_from_honorific'] is not None:
        g = row['gender_from_honorific']
    
    index = find_index_if_present(idx, None, l, must_match=g)
    if index is not None:
        return index
    
    return None


def annotate_characters(input_xml_path, output_xml_path):
    
    parser = ET.XMLParser(huge_tree=True, remove_blank_text=True)
    tree = ET.parse(str(input_xml_path), parser=parser)
    book = tree.getroot()
    
    # If a raw character mention happens <= map_threshold times, it is ignored during merging
    map_threshold = 1

    # If a character's total mentions (in all forms) are <= min_occ_threshold, it is not listed as a character
    min_occ_threshold = 1

    # The minimum number of times a string must occur, to be considered as a PERSON entity
    entity_occ_threshold = 2
    
    # Define honorifics
    male_honorifics = ['mr.', 'mr', 'mister.', 'mister', 'lord', 'master', 'sir', 'sire']
    female_honorifics = ['miss.', 'miss', 'ms.', 'ms', 'mrs.', 'mrs', 'madam', 'madame', 'lady', 'dame']
    neutral_honorifics = ['dr.', 'dr', 'doctor', 'prof.', 'prof', 'professor']

    honorifics = male_honorifics + female_honorifics + neutral_honorifics

    # Find all person entities
    person_entities = book.findall('.//entity[@ner="PERSON"]')

    # Fix to include honorifics
    for p in person_entities:
        prev = p.getprevious()
        if prev is not None and prev.tag == 't':
            if prev.text.lower() in honorifics:
                p.insert(0, prev)
                prev.tail = p.text
                p.attrib['phrase'] = prev.text + ' ' + p.attrib['phrase']
    
    


    # Add entity tags for character occurrences that were not tagged as entities

    entities = book.findall('.//entity[@ner="PERSON"]')
    entity_mentions = defaultdict(lambda: 0)
    for e in entities:
        entity_mentions[' '.join([t.text for t in e.findall('.//t')])] += 1

    entity_mentions = [x for x in entity_mentions if entity_mentions[x] >= entity_occ_threshold]

    tokens = book.findall('.//t')
    token_words = [t.text for t in tokens]


    for i, mention in enumerate(entity_mentions):

        pattern = mention.split()
        indices, l = subfinder(token_words, pattern)

        non_entity_indices = list()
        for idx in indices:
            if all(tokens[idx + n].getparent().tag != 'entity' for n in range(l)):
                non_entity_indices.append(idx)
            elif all(tokens[idx + n].getparent().tag == 'entity' for n in range(l)):
                p = tokens[idx].getparent()
                for a in p.attrib:
                    del p.attrib[a]
                p.attrib['phrase'] = ' '.join([t.text for t in p.findall('t')])
                p.attrib['ner'] = 'PERSON'

        #enclose in entity tags
        for idx in non_entity_indices:
            entity = ET.Element('entity')
            tokens[idx].addprevious(entity)
            for n in range(l):
                entity.append(tokens[idx + n])
            entity.attrib['phrase'] = ' '.join([t.text for t in entity.findall('t')])
            entity.attrib['ner'] = 'PERSON'
            
    book_tokens = book.findall('.//t')
    
    male_pronouns = {'he', 'him', 'his', 'himself'}
    female_pronouns = {'she', 'her', 'hers', 'herself'}
    all_pronouns = male_pronouns.union(female_pronouns)
    
    
    first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself'}
    second_person_pronouns = {'you', 'your', 'yours', 'yourself'}
    other_pronouns = first_person_pronouns.union(second_person_pronouns)
    
    
    # For each entity, see if any other tokens have coreference to it

    coref_dict = defaultdict(lambda:list())
    for t in book_tokens:
        if 'coref_tok_num_start' in t.attrib:
            s, e = int(t.attrib['coref_tok_num_start']), int(t.attrib['coref_tok_num_end'])

            for idx in range(s, e):
                p = book_tokens[idx].getparent()
                if p.tag == 'entity' and p.attrib['ner'] == 'PERSON':
                    coref_dict[idx].append(int(t.attrib['num']))
                    break
                    
    # For every person entity, list coreferences referring to it

    male_counts = defaultdict(lambda: 0)
    female_counts = defaultdict(lambda: 0)
    # first_person_counts = defaultdict(lambda: 0)
    # second_person_counts = defaultdict(lambda: 0)

    male_coref_tokens = defaultdict(lambda: list())
    female_coref_tokens = defaultdict(lambda: list())
    first_person_coref_tokens = defaultdict(lambda: list())
    second_person_coref_tokens = defaultdict(lambda: list())


    person_entities = book.findall('.//entity[@ner="PERSON"]')

    # Remove person entities with < entity_occ_threshold occurrences
    entity_texts = [' '.join([x.text for x in entity.findall('.//t')]).lower() for entity in person_entities]
    c = Counter(entity_texts)
    new_person_entities = list()
    for i in range(len(person_entities)):
        if c[entity_texts[i]] >= entity_occ_threshold:
            new_person_entities.append(person_entities[i])
    person_entities = new_person_entities

    for entity in person_entities:
        toks = entity.findall('.//t')

        coreference_words = set()

        for t in toks:

            t_num = int(t.attrib['num'])

            if t_num in coref_dict:

                for coref_tok in coref_dict[t_num]:
                    coreference_words.add(coref_tok)

        merged = merge_consecutive(sorted([(idx, book_tokens[idx].text) for idx in coreference_words]))

        entity_referred_to = ' '.join([x.text for x in toks])
        entity_first_token = int(toks[0].attrib['num'])
        for idx_list, tok_list in merged:
            for i in idx_list:
                p = book_tokens[i].getparent()
                if p.tag == 'entity' and p.attrib['ner'] == 'PERSON':
                    toks = p.findall('.//t')
                    entity_referred_to = ' '.join([x.text for x in toks])
                    entity_first_token = i
                    break

            tok_text = tok_list.lower()
            if tok_text in male_pronouns:
                male_counts[entity_referred_to.lower()] += 1
                male_coref_tokens[entity_first_token].append(i)
            elif tok_text in female_pronouns:
                female_counts[entity_referred_to.lower()] += 1
                female_coref_tokens[entity_first_token].append(i)
            elif tok_text in first_person_pronouns:
                first_person_coref_tokens[entity_first_token].append(i)
            elif tok_text in second_person_pronouns:
                second_person_coref_tokens[entity_first_token].append(i)
                
    
    
    person_names = [' '.join([t.text for t in p.findall('.//t')]).lower() for p in person_entities]


    l = list()
    features = ['honorific', 'generic_name', 'first_name', 'first_initial', 'middle_name', 'middle_initial', 'last_name', 'last_initial']


    for name in person_names:
        d = parse_name(name, honorifics)
        l.append([name] + [d[x] for x in features])

    df = pd.DataFrame(l, columns=['name'] + features)

    # Number of times the raw name occurs
    df['name_count'] = df.groupby('name')['name'].transform('count')
    
    df['gender_from_coref'] = df['name'].apply(lambda x: get_gender_from_coref(x, male_counts, female_counts))
    
    
    # Remove trailing period in honorific
    df['honorific'] = df['honorific'].apply(lambda x: x[:-1] if x is not None and x[-1] == '.' else x)

    condition1 = ~df['first_name'].isna()
    condition2 = ~df['last_name'].isna()
    condition = condition1 | condition2
    df['is_full_name'] = condition

    # Infer gender from first name and honorific
    d = Detector(case_sensitive=False)

    s = set()
    for idx, row in df.iterrows():
        if row['first_name'] is not None:
            s.add(row['first_name'])
            continue
        if row['generic_name'] is not None:
            if len(row['generic_name'].split()) == 1:
                s.add(row['generic_name'])
                continue

    gender_dict = {x: d.get_gender(x) for x in s if x is not None}

    l = list()
    for idx, row in df.iterrows():
        name = None
        if row['first_name'] is not None:
            name = row['first_name']
        elif row['generic_name'] is not None:
            name = row['generic_name']
        if name is not None and name in gender_dict:
            l.append(parse_gender(gender_dict[name]))
        else:
            l.append(None)

    df['gender_from_name'] = l

    df['gender_from_honorific'] = df['honorific'].apply(lambda x: get_gender_from_honorific(x, male_honorifics, female_honorifics))
    
    
    # Apply honorific to all occurrences of full name
    name_ref_full = defaultdict(lambda: list())
    condition1 = ~df['first_name'].isna()
    condition2 = ~df['last_name'].isna()
    condition3 = ~df['honorific'].isna()
    for idx, row in df[condition1 & condition2 & condition3].iterrows():
        if row['gender_from_honorific'] == row['gender_from_name'] == row['gender_from_coref']:
            name_ref_full[(row['first_name'], row['last_name'])].append((idx, row['honorific'], row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']))
        
    
    
    
    # For special cases such as Mrs. John Wilson
    name_ref_mrs = defaultdict(lambda: list())
    for fn, ln in name_ref_full:
        honorific_set = set([elem[1] for elem in name_ref_full[(fn, ln)]])
        if any([x in honorific_set for x in male_honorifics]) and 'mrs' in honorific_set:
            remove_indices = [idx for idx, elem in enumerate(name_ref_full[(fn, ln)]) if elem[1] == 'mrs']
            name_ref_full[(fn, ln)] = [x for idx, x in enumerate(name_ref_full[(fn, ln)]) if idx not in remove_indices]
            name_ref_mrs[fn] = [[x[0], x[2], x[3], x[4]] for idx, x in enumerate(name_ref_full[(fn, ln)]) if idx in remove_indices]
            name_ref_mrs[ln] = [[x[0], x[2], x[3], x[4]] for idx, x in enumerate(name_ref_full[(fn, ln)]) if idx in remove_indices]
            
            
            
    # Apply to all occurrences of full name
    condition1 = ~df['first_name'].isna()
    condition2 = ~df['last_name'].isna()
    condition3 = df['honorific'].isna()
    for idx, row in df[condition1 & condition2 & condition3].iterrows():
        # Map to closest occurrence with honorific
        if (row['first_name'], row['last_name']) in name_ref_full:
            l = name_ref_full[(row['first_name'], row['last_name'])]

            hon, gh = None, None
            for index, honorific, gender_h, _, _ in l[::-1]:
                if index < idx:
                    hon, gh = honorific, gender_h
                    break
            if hon is None:
                for index, honorific, gender_h, _, _ in l:
                    if index > idx:
                        hon, gh = honorific, gender_h
                        break
            if hon is None:
                continue

            df.loc[idx, 'honorific'] = hon
            df.loc[idx, 'gender_from_honorific'] = gender_h


            
    # Creating reference dictionary to resolve generic names
    name_ref = defaultdict(lambda: list())
    # First and last names
    for idx, row in df[(df['is_full_name']) & (df['name_count'] > map_threshold)].iterrows():
        if row['first_name'] is not None:
            if row['gender_from_honorific'] is None or row['gender_from_name'] is None or row['gender_from_honorific'] == row['gender_from_name']:
                name_ref[row['first_name']].append([idx, row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']])
            else:
                name_ref_mrs[row['first_name']].append([idx, row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']])

        if row['last_name'] is not None:
            if row['first_name'] is None:
                name_ref[row['last_name']].append([idx, row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']])
            elif row['gender_from_honorific'] is None or row['gender_from_name'] is None or row['gender_from_honorific'] == row['gender_from_name']:
                name_ref[row['last_name']].append([idx, row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']])
            else:
                name_ref_mrs[row['last_name']].append([idx, row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']])

    # Generic names
    name_ref_2 = defaultdict(lambda: list())
    for idx, row in df[(~df['is_full_name']) & (df['name_count'] > map_threshold)].iterrows():
        if row['generic_name'] is not None:
            if row['generic_name'] not in name_ref:
                name_ref_2[row['generic_name']].append([idx, row['gender_from_honorific'], row['gender_from_name'], row['gender_from_coref']])
    for elem in name_ref_2:
        name_ref[elem] = sorted(name_ref[elem] + name_ref_2[elem])
        
        
    # Modify a copy of the original dataframe
    df2 = df.copy()

    for idx, row in df2.iterrows():
        # Find next closest occurrence where generic name = first name or generic name = last name
        if row['is_full_name']:
            continue

        if row['generic_name'] is not None:
            if row['generic_name'] in name_ref:
                l = name_ref[row['generic_name']]

                elem = find_matching_index(idx, row, l)

                # Special case: Mrs
                if elem is None:
                    if row['honorific'] == 'mrs' and row['generic_name'] in name_ref_mrs:
                        l = name_ref_mrs[row['generic_name']]
                        for index, g_hon, g_name, g_coref in l:
                            if index > idx:
                                elem = index
                                break
                        if elem is None:
                            for index, g_hon, g_name, g_coref in l[::-1]:
                                if index < idx:
                                    elem = index
                                    break

                if elem is None:
                    continue

                df2.loc[idx, 'first_name'] = df.loc[elem].first_name
                df2.loc[idx, 'last_name'] = df.loc[elem].last_name

                if df.loc[elem].gender_from_honorific is not None:
                    df2.loc[idx, 'gender_from_honorific'] = df.loc[elem].gender_from_honorific
                elif df2.loc[idx, 'gender_from_honorific'] is not None and df.loc[elem].gender_from_honorific is None:
                    df.loc[elem, 'gender_from_honorific'] = df2.loc[idx, 'gender_from_honorific']

                if df.loc[elem].gender_from_name is not None:
                    df2.loc[idx, 'gender_from_name'] = df.loc[elem].gender_from_name
                elif df2.loc[idx, 'gender_from_name'] is not None and df.loc[elem].gender_from_name is None:
                    df.loc[elem, 'gender_from_name'] = df2.loc[idx, 'gender_from_name']

                if df.loc[elem].gender_from_coref is not None:
                    df2.loc[idx, 'gender_from_coref'] = df.loc[elem].gender_from_coref
                elif df2.loc[idx, 'gender_from_coref'] is not None and df.loc[elem].gender_from_coref is None:
                    df.loc[elem, 'gender_from_coref'] = df2.loc[idx, 'gender_from_coref']

    # Remove generic name if first name is present
    for idx, row in df2.iterrows():
        if row.first_name is not None:
            df2.loc[idx, 'generic_name'] = None
        if row.last_name is not None:
            df2.loc[idx, 'generic_name'] = None


    df2['gender'] = df2['gender_from_name']

    for idx, row in df2.iterrows():
        if row['gender_from_coref'] is not None:
            df2.loc[idx, 'gender'] = row['gender_from_coref']

    for idx, row in df2.iterrows():
        if row['gender_from_honorific'] is not None:
            df2.loc[idx, 'gender'] = row['gender_from_honorific']
    
    
    df2['count'] = 1
    
    
    if len(df2) == 0:
         # Add tags for characters in metadata
        m = book.find('meta')

        characters = ET.SubElement(m, "characters")
        
        with open(output_xml_path, 'wb') as f:
            f.write(ET.tostring(book, pretty_print=True))
        
        return
    
    
    character_list = df2.fillna('').pivot_table(values='count', index=['last_name', 'first_name','generic_name', 'gender'], aggfunc='sum').sort_values(['count', 'last_name', 'first_name'], ascending=False).reset_index()
    
    # Remove duplicates
    for idx, row in character_list.iterrows():
        if len(row['gender']) == 0:
            condition1 = character_list['last_name'] == row['last_name']
            condition2 = character_list['first_name'] == row['first_name']
            condition3 = character_list['generic_name'] == row['generic_name']
            condition4 = ~character_list['gender'].apply(lambda x: len(x) == 0)
            condition = condition1 & condition2 & condition3 & condition4
            tmp = character_list[condition].sort_values(by='count', ascending=False)
            if len(tmp) == 0:
                continue
            g = tmp.iloc[0]['gender']

            if len(row['last_name']) > 0:
                condition1 = df2['last_name'] == row['last_name']
            else:
                condition1 = df2['last_name'].isna()

            if len(row['first_name']) > 0:
                condition2 = df2['first_name'] == row['first_name']
            else:
                condition2 = df2['first_name'].isna()

            if len(row['generic_name']) > 0:
                condition3 = df2['generic_name'] == row['generic_name']
            else:
                condition3 = df2['generic_name'].isna()

            condition4 = df2['gender'].isna()
            condition = condition1 & condition2 & condition3 & condition4
            tmp = df2[condition]
            for idx2, row2 in tmp.iterrows():
                df2.loc[idx2, 'gender'] = g

    character_list = df2.fillna('').pivot_table(values='count', index=['last_name', 'first_name','generic_name', 'gender'], aggfunc='sum').sort_values(['count', 'last_name', 'first_name'], ascending=False).reset_index()
    
    
    character_list = character_list[character_list['count'] > min_occ_threshold]
    
    
    name_to_index = dict()
    index_to_name = dict()

    for idx, row in character_list.iterrows():
        ln = row['last_name'] if len(row['last_name']) > 0 else None
        fn = row['first_name'] if len(row['first_name']) > 0 else None
        gn = row['generic_name'] if len(row['generic_name']) > 0 else None
        gender = row['gender'] if len(row['gender']) > 0 else None

        name_to_index[(ln, fn, gn, gender)] = idx
        index_to_name[idx] = (ln, fn, gn, gender)


    character_name_counts = defaultdict(lambda: defaultdict(lambda: 0))
    character_coref_counts = defaultdict(lambda: defaultdict(lambda: 0))
    character_first_person_counts = defaultdict(lambda: defaultdict(lambda: 0))
    character_second_person_counts = defaultdict(lambda: defaultdict(lambda: 0))

    # Annotate entity tags with character ids
    for idx, row in df2.iterrows():
        #name = row['name']
        name = ' '.join([t.text for t in person_entities[idx].findall('.//t')])

        ln, fn, gn, gender = row['last_name'], row['first_name'], row['generic_name'], row['gender']

        if (ln, fn, gn, gender) not in name_to_index:
            continue

        char_idx = name_to_index[(ln, fn, gn, gender)]

        person_entities[idx].attrib['character'] = str(char_idx)

        character_name_counts[char_idx][name] += 1

        entity_first_token = int(person_entities[idx].find('.//t').attrib['num'])
        if gender == 'female':
            use_dict = female_coref_tokens
        elif gender == 'male':
            use_dict = male_coref_tokens
        if gender is not None:
            for tok_num in use_dict[entity_first_token]:
                book_tokens[tok_num].attrib['coref_character'] = str(char_idx)
                character_coref_counts[char_idx][name] += 1

        for tok_num in first_person_coref_tokens[entity_first_token]:
            book_tokens[tok_num].attrib['coref_character'] = str(char_idx)
            character_first_person_counts[char_idx][name] += 1
        for tok_num in second_person_coref_tokens[entity_first_token]:
            book_tokens[tok_num].attrib['coref_character'] = str(char_idx)
            character_second_person_counts[char_idx][name] += 1


    # Add tags for characters in metadata
    m = book.find('meta')

    characters = ET.SubElement(m, "characters")

    for idx in index_to_name:
        entered_loop = True
        c = ET.SubElement(characters, "character")
        c.attrib['id'] = str(idx)
        ln, fn, gn, g = index_to_name[idx]
        if fn is not None:
            c.attrib['first_name'] = fn
        if ln is not None:
            c.attrib['last_name'] = ln
        if gn is not None:
            c.attrib['generic_name'] = gn
        if g is not None:
            c.attrib['gender'] = g

        total_count = 0
        total_gendered_coref_count = 0
        total_first_person_coref_count = 0
        total_second_person_coref_count = 0

        for x in sorted(character_name_counts[idx].items(), key=lambda item: -item[1]):
            mention = x[0]

            n = ET.SubElement(c, "name")
            n.text = mention

            n.attrib['count'] = str(character_name_counts[idx][mention])
            total_count += character_name_counts[idx][mention]

            n.attrib['gendered_coref_count'] = str(character_coref_counts[idx][mention])
            total_gendered_coref_count += character_coref_counts[idx][mention]

            n.attrib['first_person_coref_count'] = str(character_first_person_counts[idx][mention])
            total_first_person_coref_count += character_first_person_counts[idx][mention]

            n.attrib['second_person_coref_count'] = str(character_second_person_counts[idx][mention])
            total_second_person_coref_count += character_second_person_counts[idx][mention]


        c.attrib['count'] = str(total_count)
        c.attrib['gendered_coref_count'] = str(total_gendered_coref_count)
        c.attrib['first_person_coref_count'] = str(total_first_person_coref_count)
        c.attrib['second_person_coref_count'] = str(total_second_person_coref_count)

        
    with open(output_xml_path, 'wb') as f:
        f.write(ET.tostring(book, pretty_print=True))