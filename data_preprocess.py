import argparse
from lib2to3.pgen2 import token
import contractions
from spellchecker import SpellChecker
spell = SpellChecker()
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from string import punctuation




from utils import *

def penn_to_wn(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

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

    #preprocessing step3: POS tagging and lemmatization
    lemma_tokens = []
    after_tagging = nltk.pos_tag(tokens) #pos_tag works optimally with truecase
    for word, tag in after_tagging:
        wn_tag = penn_to_wn(tag)
        if wn_tag == None:
            lemma_tokens.append(word)
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        lemma_tokens.append(lemma)
    # print(lemma_tokens)

    #preprocessing step4: removing punctuation and stopwords, change to lowercase
    stop_words = stopwords.words('english')
    punctuation = list(punctuation)
    cleaned_tokens = [token for token in lemma_tokens 
                        if token.lower() not in stop_words 
                        and token not in punctuation]
    # print(cleaned_tokens)

    #preprocessing step5: removing digits
    cleaned_tokens = [token.lower() for token in cleaned_tokens 
                        if token.isalpha()]

    #write to file
    ff.write(' '.join(cleaned_tokens)+'\n')


fp.close()
ff.close()
# print(misspelled_words)