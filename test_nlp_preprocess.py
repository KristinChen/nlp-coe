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
import pytest
import contextualSpellCheck
from zach_dev.src import nlp_preprocess
nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
# stopwords=stopwords.words('english')

class TestRemovePunct(object):
    
    @pytest.mark.parametrize('sentence, include, expected',[("Hi!@#$%^&*()-+=[]{};''`~ hello:\\.",[],'Hi hello'),
                                                    ('"Hi", he said.',[',','.'],'Hi, he said.'),
                                                    ('This\\is_@_t3st!',['\\','@','!'], 'This\\is@t3st!'),
                                                    ('The 10-12" is designed for back/side sleepers.',['/'],'The 1012 is designed for back/side sleepers'),
                                                    ('...keep three dots but exclude single dot.',['...'],'...keep three dots but exclude single dot'),
                                                    ('keep one dot. but remove three or more dots...',['.'],'keep one dot. but remove three or more dots'),
                                                    ("so for me - it's a A+",['+', '-'],'so for me - its a A+')])
                                                    
    def test_normal_arguments(self,sentence,include,expected):
        assert nlp_preprocess.remove_punct(sentence,include) == expected

    def test_bad_arguments(self):
        sentence = ['This','is','a','list']
        with pytest.raises(TypeError) as exception:
            nlp_preprocess.remove_punct(sentence)
        assert exception.match('Input for "sentence" should be a string')
        
    def test_bad_arguments2(self):
        sentence = 'This\\is_@_t3st!'
        include = '_'
        with pytest.raises(TypeError) as exception:
            nlp_preprocess.remove_punct(sentence, include)
        assert exception.match('Input for "include" should be a list')

    #keep .  but no ... print rule (more than three dots considered as ...) proper format? check grammatical/typo function can handle .....or not
   # \\ change this in backend. No regrex. Input \\ then only \\ be taken

class TestCleanTypos(object):

    @pytest.mark.parametrize('sentence, expected',[
                                                    ('hello I an Jack','hello I am Jack'),
                                                    ('helli I an Jack','hello I am Jack'),
                                                    ('helli I an Jack, noce to meet you','helli I an Jack, noce to meet you'),
                                                    ('h0w fAr cam thes modil g@', 'how far can this model go'),
                                                    ('i hate chewwwy' , 'I hate chewy'),
                                                    ('Interast rate is lower now', 'Interest rate is lower now')
                                                    ])
    def test_normal_arguments(self,sentence,expected):
        assert nlp_preprocess.clean_typos(sentence) == expected
# test nouns having specific meanings. test "lol", "asap"...abbreviations
# see corpus used for the pre-trained model - what nouns are included
# any better models?