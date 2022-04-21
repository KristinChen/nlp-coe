import random
random.seed(1234)

# Load dependencies -------------------
import pandas as pd
import nltk
nltk.download('punkt')
import pickle
import random
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize

# Load functions ------------------------
def to_word_idx(vocab, analyzer, sentence, MAX_MESSAGE_LEN, UNK = 0, PAD = 1):
    full_length = [vocab.get(tok, UNK) for tok in analyzer(sentence)] + [PAD] * MAX_MESSAGE_LEN
    return full_length[:MAX_MESSAGE_LEN]

def from_word_idx(vocab, word_idxs, PAD = 1):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ' '.join(reverse_vocab[idx] for idx in word_idxs if idx != PAD).strip()


# Load global parameter and preprocessed data ------------------
UNK = 0
PAD = 1
START = 2

with open('./data/x_text.pickle', 'rb') as handle:
    x_text = pickle.load(handle)

with open('./data/y_text.pickle', 'rb') as handle:
    y_text = pickle.load(handle)

# Vectorize data -------------------------
MAX_VOCAB_SIZE = 2**13
count_vec = CountVectorizer(tokenizer=casual_tokenize, max_features=MAX_VOCAB_SIZE - 3)
count_vec.fit(x_text + y_text)
analyzer = count_vec.build_analyzer()
vocab = {k: v + 3 for k, v in count_vec.vocabulary_.items()}
vocab['__unk__'] = UNK
vocab['__pad__'] = PAD
vocab['__start__'] = START

# Prepare training and testing data --------------------
x = pd.np.vstack(x_text.apply((lambda x: to_word_idx(vocab = vocab, analyzer = analyzer, MAX_MESSAGE_LEN = 30, sentence= x)))
.values)
y = pd.np.vstack(y_text.apply((lambda x: to_word_idx(vocab = vocab, analyzer = analyzer, MAX_MESSAGE_LEN = 30, sentence= x)))
.values)

all_idx = list(range(len(x)))
train_idx = set(random.sample(all_idx, int(0.8 * len(all_idx))))
test_idx = {idx for idx in all_idx if idx not in train_idx}

train_x = x[list(train_idx)]
test_x = x[list(test_idx)]
train_y = y[list(train_idx)]
test_y = y[list(test_idx)]

assert train_x.shape == train_y.shape
assert test_x.shape == test_y.shape

# Save data --------------------------------------
with open('./data/train_x', 'wb') as handle:
    pickle.dump(train_x, handle)

with open('./data/test_x', 'wb') as handle:
    pickle.dump(test_x, handle)

with open('./data/train_y', 'wb') as handle:
    pickle.dump(train_y, handle)

with open('./data/test_y', 'wb') as handle:
    pickle.dump(test_y, handle)

with open('./data/vectorizer', 'wb') as handle:
    pickle.dump(count_vec, handle)

