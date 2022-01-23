from stonybook.preprocessing.regex.regex_helper import get_best_matching_rule, get_header_attrs
import numpy as np
import pickle

def findRestartIndices(nums):
    if len(nums) == 0:
        return []
    ans = [0]
    if len(nums) == 1:
        return ans
    
    idx = 1
    while idx < len(nums):
        if nums[idx] <= nums[idx - 1]:
            ans.append(idx)
        idx += 1
    return ans

def is_consecutive(nums):
    for i in range(len(nums) - 1):
        if nums[i + 1] != nums[i] + 1:
            return False
    return True

def is_consecutive_relaxed(nums):
    count = 0
    for i in range(len(nums) - 1):
        if nums[i + 1] > nums[i] + 2:
            return False
        if nums[i + 1] > nums[i] + 1:
            count += 1
        if count > 3:
            return False
    return True

def is_increasing(arr):
    for i in range(len(arr) - 1):
        if arr[i + 1] <= arr[i]:
            return False
    return True

def check_match(seq1, nums1, seq2, nums2):
    
    if not is_consecutive(sorted(nums1 + nums2)):
        return False
    
    arr1 = [(nums1[i], seq1[i]) for i in range(len(nums1))]
    arr2 = [(nums2[i], seq2[i]) for i in range(len(nums2))]
    
    a = sorted(arr1 + arr2)
    
    a2 = [elem[1] for elem in a]
    
    return is_increasing(a2)


def make_rule_generic(string):
    l = string.split(',')
    l = [x for x in l if x not in {'punctuation', 'whitespace', 'title_upper', 'title_lower'}]
    return ','.join(l)


def convert_base_to_header_annot(book, regex_tuple, output_dir=None):
    
    all_seqs, rules, priority = regex_tuple
    
    paras = book.findall('.//p')
    
    # If more than 75% of the paragraph is all caps, convert the whole para to upper
    for idx in range(len(paras)):
        s = paras[idx].text
        alph = list(filter(str.isalpha, s))
        if len(alph) == 0:
            continue
        caps_ratio = sum(map(str.isupper, alph)) / len(alph)
        if caps_ratio > 0.75:
            paras[idx].text = s.upper()
    
    # Get matching rules for each paragraph
    matching_rules = [get_best_matching_rule(p.text.strip().strip("\"'"), all_seqs, rules, priority) for p in paras]
    
    # Get header attributes
    matched_rules = list()

    
    # Decide which headers to merge with next ones
    for idx, elem in enumerate(matching_rules):
        if elem:
            desc, number, number_text, number_type, title, rule_text = get_header_attrs(paras[idx].text.strip().strip("\"'"), elem[0])
            all_rules = elem
            matched_rules.append([idx, all_rules, desc, number, number_text, number_type, title, rule_text])
            
    if output_dir is not None:
        with open(str(output_dir/'matched_rules_2.pkl'), 'wb') as f:
            pickle.dump(matched_rules, f)
            
    word_numbers = ['roman_upper', 'roman_lower', 'numeral', 'word_number_one', 'word_number_One', 'word_number_ONE', 'word_number_first', 'word_number_First', 'word_number_FIRST']
    rule_set = {x[-1] for x in matched_rules}

    to_merge = list()
    for elem in rule_set:
        if not any(word_num in elem for word_num in word_numbers):
            continue

        matches = [(idx, x) for idx, x in enumerate(matched_rules) if x[-1] == elem]

        for idx, x in matches:
            if idx + 1 >= len(matched_rules):
                continue
            para_num = x[0]
            next_match = matched_rules[idx + 1]

            if next_match[0] == para_num + 1 \
            and 'title_upper' not in x[-1] and 'title_lower' not in x[-1] \
            and 'desc' not in next_match[-1] \
            and 'contents' not in str(next_match[-2]).lower() \
            and ['title_upper'] in next_match[1]: # or ['title_lower'] in next_match[1]):

                # Merge para_num and next
                to_merge.append(para_num)
    
    
    to_merge = sorted(list(set(to_merge)), reverse=True)
    
    if output_dir is not None:
        with open(str(output_dir/'to_merge_2.pkl'), 'wb') as f:
            pickle.dump(to_merge, f)
            
    # Merge paragraphs
    for para_num in to_merge:
        next_para_num = para_num + 1
        paras[para_num].text += '\n\n' + paras[next_para_num].text
        paras[next_para_num].getparent().remove(paras[next_para_num])
    
    
    # Get header attributes for matched rules after merging
    paras = book.findall('.//p')
    matching_rules = [get_best_matching_rule(p.text.strip().strip("\"'"), all_seqs, rules, priority) for p in paras]

    matched_rules = list()

    for idx, elem in enumerate(matching_rules):
        if elem:
            desc, number, number_text, number_type, title, rule_text = get_header_attrs(paras[idx].text.strip().strip("\"'"), elem[0])
            matched_rules.append([idx, desc, number, number_text, number_type, title, rule_text])
    
    if output_dir is not None:
        with open(str(output_dir/'matched_rules_new_2.pkl'), 'wb') as f:
            pickle.dump(matched_rules, f)
            
    # Decide which rules to keep
    
    matched_rules_new = [x[:-1] + [make_rule_generic(x[-1])] for x in matched_rules]
    rule_set = {(x[1], x[-1]) for x in matched_rules_new}


    sets_of_keep_indices = list()

    for desc_word, elem in rule_set:
        if not any(word_num in elem for word_num in word_numbers):
            continue

        matches = [(idx, x) for idx, x in enumerate(matched_rules_new) if x[-1] == elem and x[1] == desc_word]

        matched_numbers = [x[1][2] for x in matches]

        restart_indices = findRestartIndices(matched_numbers)
        restart_indices.append(len(matched_numbers))

        for i in range(len(restart_indices) - 1):

            from_idx, to_idx = restart_indices[i], restart_indices[i + 1]
            keep_indices = [x[0] for x in matches[from_idx:to_idx]]
            numbers = matched_numbers[from_idx:to_idx]

            sets_of_keep_indices.append((keep_indices, numbers))
    
    added = set()
    final_list = list()

    for idx in range(len(sets_of_keep_indices)):
        if idx in added:
            continue

        seq1, nums1 = sets_of_keep_indices[idx]
        if len(seq1) < 2:
            continue

        if is_consecutive(nums1):
            final_list.append(seq1)
            added.add(idx)

        else:

            for idx2 in range(len(sets_of_keep_indices)):
                if idx2 == idx:
                    continue
                if idx2 in added:
                    continue
                seq2, nums2 = sets_of_keep_indices[idx2]

                if check_match(seq1, nums1, seq2, nums2):
                    final_list.append(seq1)
                    final_list.append(seq2)
                    added.add(idx)
                    added.add(idx2)

        if idx not in added:
            if is_consecutive_relaxed(nums1):
                final_list.append(seq1)
                added.add(idx)

    keep_rule_set = [[matched_rules[idx] for idx in keep_indices] for keep_indices in final_list]
    keep_rule_set = sorted(keep_rule_set)   
    
    if output_dir is not None:
        with open(str(output_dir/'keep_rule_set_2.pkl'), 'wb') as f:
            pickle.dump(keep_rule_set, f)
    
    
    # If nothing matches
    matched_rule_indices = [x[0] for x in matched_rules]

    if len(keep_rule_set) == 0:
        # Keep only title_uppers if any exist
        final_rule_list = [x for x in matched_rules if x[-1] == 'title_upper']
        # Else keep as-is
        if len(final_rule_list) == 0:
            final_rule_list = [x for x in matched_rules] # if x[-1] != 'title_lower']

    else:
        final_rule_list = list()
        for rule_group in keep_rule_set:
            para_nums = [x[0] for x in rule_group]

            if len(para_nums) < 2:
                final_rule_list += rule_group

            else:

                # Check if it is TOC with nesting
                count = 0
                for i in range(len(para_nums) - 1):
                    curr_num = para_nums[i]
                    next_num = para_nums[i + 1]

                    for p_num in range(curr_num + 1, next_num):
                        if p_num in matched_rule_indices:
                            count += 1
                if count / (para_nums[-1] - para_nums[0]) > 0.75:
                    continue

                mean_len = np.mean([para_nums[i + 1] - para_nums[i] for i in range(len(para_nums) - 1)])

                if mean_len >= 4:
                    final_rule_list += rule_group
                
             
            
    
    # Add header attributes
    for elem in final_rule_list:
        idx, desc, number, number_text, number_type, title, rule_text = elem

        paras[idx].tag = 'header'
        paras[idx].set('desc', str(desc))
        paras[idx].set('number', str(number))
        paras[idx].set('number_text', str(number_text))
        paras[idx].set('number_type', str(number_type))
        paras[idx].set('title', str(title))
        paras[idx].set('rule_text', str(rule_text))
    
    
    # Merge consecutive title_uppers and title_lowers
    final_rule_list = sorted(final_rule_list)
    titles = {'title_upper', 'title_lower'}
    to_merge = list()
    for i, elem in enumerate(final_rule_list):
        idx, desc, number, number_text, number_type, title, rule_text = elem

        if i == len(final_rule_list) - 1:
            continue

        next_elem = final_rule_list[i + 1]
        if rule_text in titles \
        and next_elem[0] == idx + 1 \
        and next_elem[-1] == rule_text:
            to_merge.append(idx)

    to_merge = sorted(to_merge, reverse=True)

    for para_num in to_merge:
        next_para_num = para_num + 1
        paras[para_num].text += '\n\n' + paras[next_para_num].text
        paras[para_num].attrib['title'] += '\n\n' + paras[next_para_num].attrib['title']
        paras[next_para_num].getparent().remove(paras[next_para_num])
    
    
    return book