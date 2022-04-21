import random
random.seed(1234)

# Load dependencies -------------------
import keras
import pandas as pd
import sklearn
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
import demoji
import pickle
import re
import random
import time
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
import tensorflow as tf
import os

def to_word_idx(sentence):
    full_length = [vocab.get(tok, UNK) for tok in analyzer(sentence)] + [PAD] * MAX_MESSAGE_LEN
    return full_length[:MAX_MESSAGE_LEN]


# Model parameters -------------------------
MAX_VOCAB_SIZE = 2**13
MAX_MESSAGE_LEN = 30
EMBEDDING_SIZE = 100
CONTEXT_SIZE = 100
BATCH_SIZE = 4
DROPOUT = 0.2
LEARNING_RATE=0.005

# Tokens needed for seq2seq
UNK = 0
PAD = 1
START = 2

SUB_BATCH_SIZE = 1000

# Load preprocessed data ------------------
with open('./data/x_text.pickle', 'rb') as handle:
    x_text = pickle.load(handle)

with open('./data/y_text.pickle', 'rb') as handle:
    y_text = pickle.load(handle)

# Vectorize data -------------------------
count_vec = CountVectorizer(tokenizer=casual_tokenize, max_features=MAX_VOCAB_SIZE - 3)
count_vec.fit(x_text + y_text)
analyzer = count_vec.build_analyzer()
vocab = {k: v + 3 for k, v in count_vec.vocabulary_.items()}
vocab['__unk__'] = UNK
vocab['__pad__'] = PAD
vocab['__start__'] = START

# Prepare training and testing data --------------------
x = pd.np.vstack(x_text.apply(to_word_idx).values)
y = pd.np.vstack(y_text.apply(to_word_idx).values)

all_idx = list(range(len(x)))
train_idx = set(random.sample(all_idx, int(0.8 * len(all_idx))))
test_idx = {idx for idx in all_idx if idx not in train_idx}

train_x = x[list(train_idx)]
test_x = x[list(test_idx)]
train_y = y[list(train_idx)]
test_y = y[list(test_idx)]

assert train_x.shape == train_y.shape
assert test_x.shape == test_y.shape

with open('./data/train_x', 'wb') as handle:
    pickle.dump(train_x, handle)

with open('./data/test_x', 'wb') as handle:
    pickle.dump(test_x, handle)

with open('./data/train_y', 'wb') as handle:
    pickle.dump(train_y, handle)

with open('./data/test_y', 'wb') as handle:
    pickle.dump(test_y, handle)