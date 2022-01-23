from stonybook.preprocessing.gutenberg.main import gutenberg_preprocess
from stonybook.pipeline.corenlp.corenlp_process import corenlp_pickle
from stonybook.pipeline.spacy.spacy_process import spacy_single_pickle
from stonybook.preprocessing.hathitrust.main import hathitrust_preprocess
from stonybook.preprocessing.regex.regex_helper import generate_final_regex_rules

from pathlib import Path
import os
import multiprocessing
from functools import partial

def process_single_gutenberg(input_txt_file, output_dir, annot_lib='corenlp', regex_tuple=None):
    try:
        # Generate raw, base, header_annotated
        gutenberg_preprocess(input_txt_file, output_dir, regex_tuple=regex_tuple)

        input_txt_file = Path(input_txt_file)
        output_dir = Path(output_dir)

        book_id = input_txt_file.stem
        output_dir = output_dir / book_id

        if annot_lib == 'corenlp':
            # Generate corenlp_annotated.xml
            corenlp_pickle([output_dir])
            # Generate character_coref_annotated.xml
            character_process(output_dir, 'corenlp_annotated.xml')

        elif annot_lib == 'spacy':
            # Generate spacy_annotated.xml
            spacy_single_pickle(output_dir)
            # Generate character_coref_annotated.xml
            character_process(output_dir, 'spacy_annotated.xml')

        else:
            print('Please provide a valid annotation library name ("corenlp" or "spacy")')
    
    except Exception as e:
        print(e)
    
    
def process_single_hathitrust(input_zip_file, output_dir, annot_lib='corenlp', regex_tuple=None):
    try:
        # Generate raw, base, header_annotated
        hathitrust_preprocess(input_zip_file, output_dir, regex_tuple=regex_tuple)

        book_id = os.path.basename(input_zip_file)[:-4]
        lib_id = os.path.basename(os.path.dirname(input_zip_file))

        output_dir = Path(output_dir)
        output_dir = output_dir / lib_id / book_id


        if annot_lib == 'corenlp':
            # Generate corenlp_annotated.xml
            corenlp_pickle([output_dir])
            # Generate character_coref_annotated.xml
            character_process(output_dir, 'corenlp_annotated.xml')

        elif annot_lib == 'spacy':
            # Generate spacy_annotated.xml
            spacy_single_pickle(output_dir)
            # Generate character_coref_annotated.xml
            character_process(output_dir, 'spacy_annotated.xml')

        else:
            print('Please provide a valid annotation library name ("corenlp" or "spacy")')
    except Exception as e:
        print(e)
    
    
def process_batch_gutenberg(input_txt_file_list, output_dir, num_threads):
    regex_tuple = generate_final_regex_rules()
    
    pool = multiprocessing.Pool(processes=num_threads)
    data = pool.map(partial(process_single_gutenberg, output_dir=output_dir, regex_tuple=regex_tuple), input_txt_file_list)
    pool.close()
    pool.join()
    
    
def process_batch_hathitrust(input_zip_file_list, output_dir, num_threads):
    regex_tuple = generate_final_regex_rules()
    
    pool = multiprocessing.Pool(processes=num_threads)
    data = pool.map(partial(process_single_hathitrust, output_dir=output_dir, regex_tuple=regex_tuple), input_zip_file_list)
    pool.close()
    pool.join()
    