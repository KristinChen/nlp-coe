from asyncio.windows_events import NULL
import re
import string
import unidecode
import contractions
from textblob import Word
from num2words import num2words
import spacy
from nltk.stem import WordNetLemmatizer
from scispacy.abbreviation import AbbreviationDetector
from typing import List, Optional
import contextualSpellCheck
import emoji
import datefinder
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.language import Language
import lexnlp.extract.en.money
nlp = spacy.load('en_core_web_sm')
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
contextualSpellCheck.add_to_pipe(nlp)
stopwords = nlp.Defaults.stop_words

### Removing Punctuations
def remove_punctuations(sentence : str, 
                 include : list = None,
                 verbose: bool = False) -> str:
    """Remove punctuations. The default is to remove all except numbers and letters. Dots more than three in a row will be treated as ellipsis. 
    To include ellipsis but not dots, put ['...'] in include. 

    Parameters
    ----------
    sentence : str :
        
    From which to remove punctuations. 
        
    include : list :
        
    A list of punctuations user wants to keep. 
        
    verbose : boolean : 

    If a list of excluded punctuations should be printed out.

    Returns
    -------
    The sentence without user defined punctuations
    
    """
    if not isinstance(sentence, str):
        raise TypeError('Input for "sentence" should be a string')
    if include:
        if not isinstance(include, list):
            raise TypeError('Input for "include" should be a list')
    
        if all([re.search('[.]{3,}',i) is None for i in include]): # if user wants to keep only one dot but remove ellipsis-like punctuations
            sentence = re.sub('[.]{3,}',' ',sentence)
            processed_sentence = sentence.translate(str.maketrans('', '', "".join(set(string.punctuation).difference(set(include)))))

        if any([re.search('[.]{3,}',i) is not None for i in include]) & ('.' not in include): # if user wants to keep ellipsis-like punctuations but remove one dot
            check_sequence = [m.start() for n in [re.finditer('[.]', sentence)] for m in n]
            for i in check_sequence:
                if (i-1 not in check_sequence) & (i+1 not in check_sequence):
                    sentence = sentence[:i] + sentence[i+1:]
            include.append('.')
            processed_sentence = sentence.translate(str.maketrans('', '', "".join(set(string.punctuation).difference(set(include)))))

        processed_sentence = sentence.translate(str.maketrans('', '', "".join(set(string.punctuation).difference(set(include)))))
        if verbose:
            print("[{}] has been excluded".format("".join(set(string.punctuation).difference(set(include)))))
    else:
        processed_sentence = sentence.translate(str.maketrans('', '', "".join(string.punctuation)))
        if verbose:
            print("[{}] has been excluded".format("".join(string.punctuation)))
    return re.sub('\s{2,}', ' ', processed_sentence).strip()



### Extracting the root words (lemmatization?) - might take long time to run
def lemmatize(sentence: str) -> str:
    """Lemmatization

    Parameters
    ----------
    sentence : str
        
    Sentence from which user wants to convert the words into root format. 
        
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
def lower_case(sentence : str):
    """
    Transform all letters to lower cases.

    Parameters
    ----------
    sentence :
    The sentence to be transformed to lower case.

    Returns
    -------
    A sentence transformed to lower case.
    
    """
    return sentence.lower()

### Removing Stop words
def remove_stopwords(sentence : str, add_stopwords : List = None, exclude_stopwords: List = None):
    """
    Remove stopwords 

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
    sentence = expand_contractions(sentence)
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
    Remove redundant spaces in the sentence.

    Parameters
    ----------
    sentence :
    The sentence user wants to remove extra spaces from.

    Returns
    -------
    The sentence without extra spaces.
    
    """
    if re.search(r'\s{1,}[^\w\s]', sentence) is not None:
        check_sequence = len([m.start() for n in [re.finditer(r'\s{1,}[^\w\s]', sentence)] for m in n])
        for i in range(check_sequence):
            j = re.search(r'\s{1,}[^\w\s]', sentence).start()
            sentence = sentence[:j] + sentence[j+1:]
    sentence = re.sub(r'\s{2,}', ' ', sentence).strip()
    return sentence

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
def convert_NumtoWords(sentence: str):
    """
    Convert numbers to words.

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
def expand_contractions(sentence :str) -> str:
    """
    Expand contractions. 

    Parameters
    ----------
    sentence : str:
    The sentence from which user wants to expand contractions
        

    Returns
    -------
    Sentence after expanding contractions.
    
    """
    return contractions.fix(sentence)

### Tokenziation
def tokenize(sentence: str, special_rule: dict= None) -> list:
    """
    Tokenize the sentence the user inputs.

    Parameters
    ----------
    sentence : str :
        
    Sentence or sentences the user inputs. 
        
    special_rule : dict :
        (Default value = None)
        A dictionary of special rules. Keys should be either "keep" or "split", indicating if user wants certain words/phrases
        to be intact or split up. Values should be lists. If key is "keep", the value should be one list containing the words/phrases user wants to
        retain after tokenization. If key is "split", the value must be multiple lists containing parts user wants to see after tokenization.
        Each nested list contains one specail case. Even there is only one rule created for "split", nested list should still be used.
        For instance, if user wants to keep "up-to-date" and "value-add", split "gimme" into "gim" and "me", and split "lemme" into
        "lem" and "me", the special_rule should be assigned as:
        {'keep':['up-to-date','value-add],'split':[['gim','me'],['lem','me']]}
        If user just wants to split "gimme" into "gim" and "me", the special_rule should be assigned as:
        {'split':[['gim','me']]}
    

    Returns
    -------
    A list of tokens
    
    """
    tokenizer = Tokenizer(nlp.vocab)
    if isinstance(special_rule,dict):
        for k, v in special_rule.items():
            for ele in v:
                if k=='split':
                    special_case = [{ORTH:ele[i]} for i in range(len(ele))]
                    tokenizer.add_special_case("".join(ele), special_case)
                if k=='keep':
                    special_case = [{ORTH:ele}]
                    tokenizer.add_special_case(ele, special_case)
    doc = tokenizer(sentence)
    tokens = [token.text for token in doc]
    return tokens
# keep + split. Make it easier

# def expand_abbreviations(sentence):
#     doc = nlp(sentence)
#     altered_tok = [tok.text for tok in doc]
#     for abrv in doc._.abbreviations:
#         altered_tok[abrv.start] = str(abrv._.long_form)

#     return(" ".join(altered_tok))

def detect_emojis(sentence: str, dedup: bool = False, verbose: bool = False):
    emoji_list = emoji.emoji_list(sentence)
    positions = []
    emojis = []
    for i in range(len(emoji_list)):
        positions.append((emoji_list[i].get('match_start'),emoji_list[i].get('match_end')))
        emojis.append(emoji_list[i].get('emoji'))
    if verbose:
        msg = "There are {} emojis detected. They are {} in position {}.".format(len(emoji_list),emojis,positions)
        print(msg)
    if dedup:
        return set(emojis)
    else:
        return emojis

def translate_emojis(sentence: str, delimiters: tuple =(""," ")):
    return emoji.demojize(sentence,delimiters)

def detect_urls(sentence: str, dedup: bool = False , formatted : bool = False):
    """
    Detect urls in the sentence.

    Parameters
    ----------
    sentence : str :
        
    Sentence the user inputs. 
        
    dedup : bool :
        (Default value = False)
    Whether the output should only contain deduplicated urls. 

    formatted : bool :
        (Default value = False)
    Whether the output should be in standard format. The standard format consists of 'https' and 'www'.

    Returns
    -------
    A list of urls
    """
    regex = re.compile(r"((?i)\b((?:https?://|w?w?w?\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))|[a-z0-9]+[.](com|net|org|eu|cn|io|biz|co|us|blog))",re.IGNORECASE)
    urls = re.findall(regex,string)
    urls = [url[0] for url in urls]
    if formatted:
        for i in range(len(urls)):
            if 'www' not in urls[i]:
                urls[i] = 'https://www.' + urls[i]
            if 'https' not in urls[i]:
                urls[i] = 'https://' + urls[i]
    if dedup:
        urls = list(set(urls))
    return urls

def detect_dates(sentence: str, format_date: bool=True, dedup: bool=False):
    dates = datefinder.find_dates(sentence)
    if format_date:
        dates = [date.strftime('%m-%d-%Y %H:%M:%S') for date in dates]
    else:
        dates = [date for date in dates]
    if dedup:
        return set(dates)
    return dates

def detect_USPhones(sentence: str, dedup: bool = False , formatted : bool = False):
    """
    Detect US phone numbers in the sentence.

    Parameters
    ----------
    sentence : str :
    Sentence the user inputs. 
        
    dedup : bool :
        (Default value = False)
    Whether the output should only contain deduplicated phone numbers. 

    formatted : bool :
        (Default value = False)
    Whether the output should be in standard format. The standard format is a 10-digit number without any hyphen or '+1' within.

    Returns
    -------
    A list of phone numbers
    """
    regex = r'([(]?[+]?[1]?[)-]?\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'
    detection = re.findall(regex, sentence)
    if formatted:
        phones = [remove_punctuations(i, include=['+']) for i in detection]
        detection = []
        for phone in phones:
            if len(phone)==11:
                detection.append(phone[1:])
            elif len(phone)==12:
                detection.append(phone[2:])
            else:
                detection.append(phone)
    if dedup:
        detection = list(set(detection))
    return detection

def detect_emails(sentence: str, dedup: bool=False):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if dedup:
        return set(re.findall(regex, sentence))
    return re.findall(regex, sentence)

def detect_citations(sentence: str, dedup: bool = False):
    regex = \
    r"\b(?!(?:Although|Also|After|Although|As|If|Though|Because|Before|By|For|Lest|Once|Since|That|Till|Unless|Until|When|Whenever|Where|Wherever|While)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:,? *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))"
    citations = re.findall(regex, sentence)
    if dedup:
        return set(citations)
    return citations

def detect_money(sentence: str, dedup: bool = False, formatted = False):
    regex1 = re.compile(r"([£$€¥]?[\s]?[0-9]+[.]?[0-9]{0,}[\s]?(?:Million|Trillion|Billion|Thousand|M|mm|tn|k|t|bn)?[\s]?)(dollars|usd|euros|eur|gbp|pounds|sterlings|yuan)?", re.IGNORECASE)    
    money1 = re.findall(regex1, sentence)
    regex2 = re.compile(r'[0-9]+[.]?[0-9]{0,}¢')
    money2 = re.findall(regex2, sentence)
    regex3 = re.compile(r'[0-9]+[.]?[0-9]{0,}')
    detection = [re.sub('\s{2,}', ' ', i).strip() for i in [ " ".join(i).strip() for i in money1]] + \
                       [i for i in money2]
    
    if formatted:
        money_dict_1 = {'£':['pounds','gbp','sterlings'],'$':['dollars','usd','bucks'],'€':['euros','eur'],'¥':['yuans','cny','rmb']}
        money_dict_2 = {k:v for lst in [*money_dict_1.values()] for k in lst for v in [*money_dict_1.keys()] if k in money_dict_1[v]}
        for i in range(len(detection)): 
            # has unit but no currency symbol
            if (re.search(r'[£$€¥]', detection[i]) is not None) and (re.search(r'(dollars|usd|bucks|euros|eur|gbp|pounds|sterlings|yuan|cny|rmb)', detection[i], re.IGNORECASE) is None):
                detection[i] = detection[i] + ' '+ money_dict_1[re.findall(r'[£$€¥]', detection[i], re.IGNORECASE)[0]][0]
            # has currency symbol but no unit
            if (re.search(r'[£$€¥]', detection[i]) is None) and (re.search(r'(dollars|usd|bucks|euros|eur|gbp|pounds|sterlings|yuan|cny|rmb)', detection[i], re.IGNORECASE) is not None):
                detection[i] = money_dict_2[re.findall(r'(dollars|usd|euros|eur|gbp|pounds|sterlings|yuan|cny|rmb)', detection[i], re.IGNORECASE)[0].lower()] + detection[i]
            # has space between currency symbol and first digit of number
            if re.search(r'[£$€¥][\s][0-9]', detection[i]) is not None:
                detection[i] = detection[i][0] + detection[i][2:]
            # has no space between last digit of number and quantity
            if re.search(r'[0-9](?:Million|Trillion|Billion|Thousand|M|mm|tn|k|t|bn)', detection[i]) is not None:
                start = re.search(r'[0-9](?:Million|Trillion|Billion|Thousand|M|mm|tn|k|t|bn)', detection[i]).span()[0]
                detection[i] = detection[i][:start+1] + ' ' + detection[i][start+1:]
    
    if dedup:
        detection = list(set(detection))
        
    detection = [i.upper() if formatted else i for i in detection if i not in re.findall(regex3, sentence)]
    return detection
# must have either 1)unit or 2)currency symbol to be detected


def split_by_sentence(sentences: str, special_rules: list = None, 
                      default_and_special: bool = False, keep_conjunction_words: bool = False):
    if special_rules:
        if len(special_rules)>1:
            regex = "|".join(special_rules)
            regex = r"\b(" + regex + r")\b"
        else:
            regex = r"\b("+ special_rules[0] + r")\b"

        if not default_and_special:
            if keep_conjunction_words:
                processed = [sent.strip() for sent in re.split(regex, sentences)]
                for i, word in enumerate(processed):
                    if word in special_rules:
                        processed[i] = word + " " + processed[i+1]
                        processed.remove(processed[i+1])
                return [sent.strip() for sent in processed]
            else:
                processed = [sent.strip() for sent in re.split(regex, sentences) if sent not in special_rules]
                try:
                    processed.remove('')
                    return processed
                except:
                    return processed
        if default_and_special:
            doc = nlp(sentences)
            sents = [sent.text for sent in doc.sents]
            if keep_conjunction_words:
                processed = [sent.strip() for sent_list in [re.split(regex, i) for i in sents] for sent in sent_list]
                for i, word in enumerate(processed):
                    if word in special_rules:
                        processed[i] = word + " " + processed[i+1]
                        processed.remove(processed[i+1])
                return [sent.strip() for sent in processed]
            else:
                return [sent.strip() for sent_list in [re.split(regex, i) for i in sents] for sent in sent_list if sent not in special_rules]        
    else:
        doc = nlp(sentences)
        return [sent.text for sent in doc.sents]

# def standard_preprocess(sentence, stopword=stopwords):
#     """

#     Parameters
#     ----------
#     sentence :
        
#     stopword :
#         (Default value = stopwords)

#     Returns
#     -------

    
#     """
#     sentence = remove_punct(sentence)
#     sentence = lower_case(sentence)
#     sentence = remove_stopwords(sentence, stopword)
#     sentence = remove_spaces(sentence)
#     return sentence
