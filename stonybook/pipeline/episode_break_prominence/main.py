import os
from pathlib import Path
from stonybook.pipeline.episode_break_prominence.add_break_prominence import add_episode_break_prominence

def break_prominence_process(book_dir, input_filename='character_coref_annotated.xml'):
    book_dir = Path(book_dir)
    input_xml_path = book_dir / input_filename
    output_xml = 'break_prominence_character_coref_annotated.xml'
    output_xml_path = book_dir / output_xml
    
    if not os.path.exists(str(output_xml_path)):
        add_episode_break_prominence(input_xml_path, book_dir, output_xml)