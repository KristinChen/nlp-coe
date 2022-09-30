from asyncio.windows_events import NULL
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unidecode
import contractions
from textblob import Word
import inflect
import spacy
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
import contextualSpellCheck
nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
# stopwords=stopwords.words('english')

### Removing Punctuations
def remove_punct(sentence : str, 
                 include : list = None) -> str:
    """
    Remove punctuations. The default is to remove all except numbers, letters. 

    Parameters
    ----------
    sentence : str : 
    From which to remove punctuations.
        
    include : list :
    A list of punctuations user wants to keep.     

    Returns
    -------
    The sentence without all or user defined punctuations. Print out the punctuations that were removed.
    """
    if not isinstance(sentence, str):
        raise TypeError('Input for "sentence" should be a string')
    if not isinstance(include, list):
        raise TypeError('Input for "include" should be a list')
        
    if all([re.search('[.]{3,}',i) is None for i in include]): # if user wants to keep only one dot but remove ellipsis-like punctuations
        sentence = re.sub('[.]{3,}','',sentence)
        processed_sentence = sentence.translate(str.maketrans('', '', "".join(set(string.punctuation).difference(set(include)))))
   
    if any([re.search('[.]{3,}',i) is not None for i in include]) & ('.' not in include): # if user wants to keep ellipsis-like punctuations but remove one dot
        check_sequence = [m.start() for n in [re.finditer('[.]', sentence)] for m in n]
        for i in check_sequence:
            if (i-1 not in check_sequence) & (i+1 not in check_sequence):
                sentence = sentence[:i] + sentence[i+1:]
        include.append('.')
        processed_sentence = sentence.translate(str.maketrans('', '', "".join(set(string.punctuation).difference(set(include)))))
        
    processed_sentence = sentence.translate(str.maketrans('', '', "".join(set(string.punctuation).difference(set(include)))))
    print("[{}] has been excluded".format("".join(set(string.punctuation).difference(include))))
    return processed_sentence



### Extracting the root words (lemmatization?) - might take long time to run
def root_words(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    doc = nlp(sentence)
    lemm_sent = []
    for token in doc:
        lemm_sent.append(token.lemma_)
    return ' '.join(lemm_sent)

### Removing Accents - might take long time to run
def remove_accents(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    return unidecode.unidecode(sentence)

### Converting to Lower case
def to_lower(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    return sentence.lower()

### Removing Stop words
def remove_stopwords(sentence, stopwords=stopwords):
    """

    Parameters
    ----------
    sentence :
        
    stopwords :
         (Default value = stopwords)

    Returns
    -------

    """
    return " ".join([word for word in sentence.split() if word not in stopwords])

### Removing Extra Spaces
def remove_spaces(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    return re.sub('\s{2,}', ' ', sentence).strip()

### Clean typos - might take long time to run
def clean_typos(sentence : str,
                show_typos: bool = False) -> str: 
# ultimately it's good to allow for user to pass customized param, in dict, key: false word, value: to what is correct
    """
    Clean typos within one sentence

    Parameters
    ----------
    sentence : str :
    The sentence on which user wants to clean typos

    show_typos: bool:
    Whether the function should print out what typos are corrected with what   

    Returns
    -------
    A sentence of type "str"

    """
    # correct_sent = ''
    # for word in sentence.split():
    #     correct_sent += Word(word).correct() + ' '
    # return correct_sent.strip()
    if not isinstance(sentence, str):
        raise TypeError('Input for "sentence" should be a string')
    doc = nlp(sentence)
    if show_typos:
        print(doc._.suggestions_spellCheck)
    return doc._.outcome_spellCheck

### number handling (1 -> one) - might take long time to run
def number_translate(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    p = inflect.engine()
    sent_split = sentence.split()
    for i, word in enumerate(sent_split):
        if word.isdigit() and int(word) < 101:
            sent_split[i] = p.number_to_words(word)
    return ' '.join(sent_split)

### Expand Contractions (don't -> do not) - might take long time to run
def expand_contractions(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    return contractions.fix(sentence)

### Tokenziation
def tokenize(sentence):
    """

    Parameters
    ----------
    sentence :
        

    Returns
    -------

    """
    return word_tokenize(sentence)

def standard_preprocess(sentence, stopword=stopwords):
    """

    Parameters
    ----------
    sentence :
        
    stopword :
         (Default value = stopwords)

    Returns
    -------

    """
    sentence = remove_punct(sentence)
    sentence = to_lower(sentence)
    sentence = remove_stopwords(sentence, stopword)
    sentence = remove_spaces(sentence)
    return sentence
