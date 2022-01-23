# StonyBook

StonyBook is a research effort to computationally analyze large corpora of novels, to develop new natural language processing (NLP) tools suitable for working with long texts, new data science/visualization techniques to better understand the historical/cultural evolution of what we read and write.

This repository contains code for the Stony Book annotation pipeline. Our analysis artifacts are available at https://stonybook.org/.


## Installation

* Optional but recommended: Create a new Anaconda environment. Install Anaconda, and create a new environment:

```
conda create --name stonybook_env python=3.7
conda activate stonybook_env
```
* Download the StonyBook package

```
git clone https://github.com/sbu-dsl/stonybook.git
```

* Install the package

```
pip install ./stonybook
```

## Usage

In each of the following, the output directory will contain a final annotated XML named `character_coref_annotated.xml`.


### Project Gutenberg books

StonyBook can process single or multiple Project Gutenberg text files.

#### Single book
For example, to process 'Oliver Twist' (Gutenberg ID 730)
```python
from stonybook.process import process_single_gutenberg
input_txt_file = 'path/to/gutenberg/data/730.txt'
output_dir = 'path/to/output/dir'

# To annotate using CoreNLP
process_single_gutenberg(input_txt_file, output_dir, annot_lib='corenlp')

# To annotate using SpaCy
process_single_gutenberg(input_txt_file, output_dir, annot_lib='spacy')
```

#### Multiple books
```python
from stonybook.process import process_batch_gutenberg
input_txt_files = ['path/to/gutenberg/data/730.txt', 'path/to/gutenberg/data/64317.txt', 'path/to/gutenberg/data/37106.txt']
output_dir = 'path/to/output/dir'

# To annotate using CoreNLP
process_batch_gutenberg(input_txt_files, output_dir, annot_lib='corenlp')

# To annotate using SpaCy
process_batch_gutenberg(input_txt_files, output_dir, annot_lib='spacy')
```

### HathiTrust books

StonyBook can process single or multiple HathiTrust zip files.

#### Single book
For example, to process 'The last of the Mohicans' (HathiTrust ID mdp.39015063553054)
```python
from stonybook.process import process_single_hathitrust
input_zip_file = 'path/to/hathitrust/data/mdp/39015063553054.zip'
output_dir = 'path/to/output/dir'

# To annotate using CoreNLP
process_single_hathitrust(input_zip_file, output_dir, annot_lib='corenlp')

# To annotate using SpaCy
process_single_hathitrust(input_zip_file, output_dir, annot_lib='spacy')
```

#### Multiple books
```python
from stonybook.process import process_batch_hathitrust
input_zip_files = ['path/to/hathitrust/data/mdp/39015063553054.zip', 'path/to/hathitrust/data/ucm/5324201722.zip', 'path/to/hathitrust/data/coo/31924064979440.zip']
output_dir = 'path/to/output/dir'

# To annotate using CoreNLP
process_batch_hathitrust(input_zip_files, output_dir, annot_lib='corenlp')

# To annotate using SpaCy
process_batch_hathitrust(input_zip_files, output_dir, annot_lib='spacy')
```

