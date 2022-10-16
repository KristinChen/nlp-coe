from asyncio.windows_events import NULL
import re
import string
import unidecode
import contractions
from textblob import Word
from num2words import num2words
import spacy
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
import contextualSpellCheck
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
stopwords = nlp.Defaults.stop_words

### Removing Punctuations
def remove_punctuation(sentence : str, 
                 include : list = None) -> str:
    """Remove punctuations. The default is to remove all except numbers, letters.

    Parameters
    ----------
    sentence : str :
        
    From which to remove punctuations. :
        
    include : list :
        
    A list of punctuations user wants to keep. :
        
    sentence : str :
        
    include : list :
         (Default value = None)

    Returns
    -------

    
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
def root_words(sentence: str) -> str:
    """Lemmatization

    Parameters
    ----------
    sentence : str
        
    Sentence from which user wants to convert the words into root format. :
        
    sentence: str :
        

    Returns
    -------

    
    """
    doc = nlp(sentence)
    lemm_sent = []
    for token in doc:
        lemm_sent.append(token.lemma_)
    return ' '.join(lemm_sent)
    # need separate method to handle punction. Do not assume order

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
def to_lower(sentence : str):
    """

    Parameters
    ----------
    sentence :
    The sentence to be transformed to lower cases

    Returns
    -------
    A sentence transformed to lower cases.
    
    """
    return sentence.lower()

### Removing Stop words
def remove_stopwords(sentence : str, add_stopwords : List = None, exclude_stopwords: List = None):
    """

    Parameters
    ----------
    sentence: str
    The sentence from which user wants to remove stopwords
    
    add_stopwords :
         (Default value = None)
    User defined list of stopwords to be added to Spacy default stopwords
    exclude_stopwords :
         (Default value = None)
    User defined list of stopwords to be excluded from Spacy default stopwords

    Returns
    -------
    A sentence without stopwords
    
    """
    stopwords = list(nlp.Defaults.stop_words)
    if add_stopwords:
        stopwords = stopwords + add_stopwords
    if exclude_stopwords:
        stopwords = [stopword for stopword in stopwords if stopword not in exclude_stopwords]

    sentence = sentence.lower()
    # make sure stopwords that are followed by punctuations will be removed as well
    processed_sentence = " ".join([word for word in re.findall(r'[\w]+|[^\s\w]+', sentence) if word not in stopwords])

    ## all punctuations between letters will be remained in same format (no space between)
    ## For instnace, 'me@outlook.com' and 'user-friendly'.
    check_sequence = len([m.start() for n in [re.finditer(r'\s[^\s\w]\s', processed_sentence)] for m in n])
    for i in range(check_sequence):
        j = re.search(r'\s[^\s\w]\s', processed_sentence).start()
        processed_sentence = processed_sentence[:j] + processed_sentence[j+1] + processed_sentence[j+3:]
    return processed_sentence

## Those punctuations that end the sentence will have a space between them with the last letter

### Removing Extra Spaces
def remove_spaces(sentence : str):
    """

    Parameters
    ----------
    sentence :
    The sentence user wants to remove extra spaces from.

    Returns
    -------
    The sentence without extra spaces.
    
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
## cannot handle negative number or float
def num_to_words(sentence: str):
    """

    Parameters
    ----------
    sentence : str:
    The sentence in which user wants to convert numbers to words
        
    Returns
    -------
    Sentence with numbers converted to words
    
    """
    check_sequence = len([m.start() for n in [re.finditer(r'\-?[\d]{1,}\.?[\d]{0,}', sentence)] for m in n])
    for i in range(check_sequence):
        j = re.search(r'\-?[\d]{1,}\.?[\d]{0,}', sentence).span()
        if sentence[j[0]] != '-':
            sentence = sentence[:j[0]] + num2words(sentence[j[0]:j[1]]) + sentence[j[1]:]
        elif sentence[j[0]] == '-':
            try:
                sentence = sentence[:j[0]] + 'negative ' + num2words(sentence[j[0]:j[1]]).replace('minus','') + sentence[j[1]:]
            except:
                sentence = sentence[:j[0]] + 'negative' + num2words(sentence[j[0]:j[1]])  + sentence[j[1]:]

    return re.sub('\s{2,}', ' ', sentence).strip()

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
def tokenize(sentence: str, special_rule: dict= None, split_by_sentence: bool = False) -> list:
    """Tokenize the sentence/sentences the user inputs.

    Parameters
    ----------
    sentence : str :
        
    Sentence or sentences the user inputs. :
        
    special_rule : dict :
        (Default value = None)
        A dictionary of special rules. Keys should be either "merge" or "split", indicating if user wants certain words/phrases
        to be intact or split up. Values should be lists. If key is "merge", the value should be one list containing the words/phrases user wants to
        retain after tokenization. If key is "split", the value must be multiple lists containing parts user wants to see after tokenization.
        Each nested list contains one specail case. Even there is only one rule created for "split", nested list should still be used.
        For instance, if user wants to keep "up-to-date" and "value-add", split "gimme" into "gim" and "me", and split "lemme" into
        "lem" and "me", the special_rule should be assigned as:
        {'merge':['up-to-date','value-add],'split':[['gim','me'],['lem','me']]}
        If user just wants to split "gimme" into "gim" and "me", the special_rule should be assigned as:
        {'split':[['gim','me']]}
    split_by_sentence : bool :
        (Default value = False)
        Set to true is user wants to tokenize by sentences rather than words. It cannot be true if special rules exist.
    sentence: str :
        
    special_rule: dict :
         (Default value = None)
    split_by_sentence: bool :
         (Default value = False)

    Returns
    -------

    
    """
    tokenizer = Tokenizer(nlp.vocab)
    if isinstance(special_rule,dict):
        for k, v in special_rule.items():
            for ele in v:
                if k=='split':
                    special_case = [{ORTH:ele[i]} for i in range(len(ele))]
                    tokenizer.add_special_case("".join(ele), special_case)
                if k=='merge':
                    special_case = [{ORTH:ele}]
                    tokenizer.add_special_case(ele, special_case)
        if split_by_sentence:
            split_by_sentence = False
            print("Can only tokenize sentences by words if special rules apply.Changed split_by_sentence to False.")
    if split_by_sentence:
        doc = nlp(sentence)
        tokens = [token.text for token in doc.sents]
    else:
        doc = tokenizer(sentence)
        tokens = [token.text for token in doc]
    return tokens
# keep + split. Make it easier

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
