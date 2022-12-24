import nltk
nltk.download('punkt')
from ocrfixr import spellcheck
import os
import re


def ocrfixr_to_new_folder(input_unzip_folder, output_unzip_folder):

    #pages in unzipped file
    for page in os.listdir(input_unzip_folder):
        corrected_page = os.path.join(output_unzip_folder, page)
        page_path = os.path.join(input_unzip_folder, page)

        #open file check mistakes and correct -> print new file
        with open(page_path, "r", encoding = "utf-8") as open_file:
            content = open_file.read()
            sentences = nltk.sent_tokenize(content)
            fixes = {}

            #finding Phase
            for sentence in sentences:
                # (setence, {(): 1})
                fix = spellcheck(sentence, return_fixes = "T").fix()
                if len(fix[1]):
                    fixes.update(fix[1])

            # replacing Phase
            for mistake_tuple, count in fixes.items():
                mistake, fix = mistake_tuple
                #using re to only fix correct strings and not substring replacements
                mistake_pattern = re.compile(r"\b{0}\b".format(re.escape(mistake)))
                content = re.sub(mistake, fix, content)

        #writing new file
        with open(corrected_page, 'w') as f:
            f.write(content)