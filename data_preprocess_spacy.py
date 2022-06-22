import argparse
from distutils.command.clean import clean
from lib2to3.pgen2 import token
import contractions
from spellchecker import SpellChecker
spell = SpellChecker()
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from string import punctuation

import spacy
nlp = spacy.load("en_core_web_sm")

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--text_file', default='data/raw_data.txt', help='input text file')
args = parser.parse_args()

misspelled_words = set()

fp = open(args.text_file, 'r')
ff = open('data/data.txt', 'w')
for line in fp: 

    #ensure all ascii chars
    encoded_string = line.encode("ascii", "ignore")
    line = encoded_string.decode()

    #preprocessing step1: expand contractions
    corrected = contractions.fix(line, slang=True)
    # print(corrected)

    #preprocessing step2: correct spellings
    tokens = word_tokenize(corrected)
    misspellings = spell.unknown(tokens)
    if len(misspellings) > 0:
        # print(misspellings)
        # have spelling error
        misspelled_words.update(misspellings)
        for (idx, i) in enumerate(tokens):
            if i.lower() in misspellings:
            #misspellings only contain lowercase letters.
                # print(tokens,i)
                tokens[idx] = spell.correction(i)

    #spacy
    combined = ' '.join(tokens)
    doc = nlp(combined)
    cleaned_tokens = []
    for token in doc:
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
        if token.is_alpha and not token.is_stop:
            cleaned_tokens.append(token.lemma_)
    #write to file
    ff.write(' '.join(cleaned_tokens)+'\n')


fp.close()
ff.close()
# print(misspelled_words)