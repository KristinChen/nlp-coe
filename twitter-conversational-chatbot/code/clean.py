# Load dependencies -------------------
import pandas as pd
import nltk
import pickle
import re
import os

# Load global parameters and functions ----------------------
with open(r'./data/Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
    Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

def sn_replace(match):
    _sn = match.group(2).lower()
    if not _sn.isnumeric():
        # This is a company screen name
        return match.group(1) + match.group(2)
    return ' @__sn__'


def replace_ID(text):
    try:
        if isinstance(text, str):
            sn_re = re.compile('(\W@|^@)([a-zA-Z0-9_]+)')
            return sn_re.sub(sn_replace, text)
        else:
            raise ValueError()
    except ValueError:
        pass


def convert_emojis_to_word(text, emoji_dict = Emoji_Dict):
    try:
        if isinstance(text, str):
            for emot in emoji_dict:
                text = re.sub(r'(' + emot + ')', "".join(emoji_dict[emot].replace(":", " ")), text)
            return text
        else:
            raise ValueError()
    except ValueError:
        pass


def replace_URL(text):
    try:
        if isinstance(text, str):
            pattern = r"http\S+|https\S+|www\S+"
            return re.sub(pattern, "__URL__", str(text))
        else:
            raise ValueError()
    except ValueError:
        pass


def clean_text(text, emoji_dict=Emoji_Dict):
    try:
        if isinstance(text, str):
            temp1 = replace_ID(text)
            temp2 = convert_emojis_to_word(temp1, emoji_dict)
            temp3 = replace_URL(temp2)
            return temp3
        else:
            raise ValueError()
    except ValueError:
        pass

# Read data ------------------
tweets = pd.read_csv('./data/twcs/twcs.csv')
first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id',
                                  right_on='in_response_to_tweet_id')
inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]


# Preprocess data -------------
x_text = inbounds_and_outbounds.text_x[0:100].apply(lambda txt: clean_text(txt))
y_text = inbounds_and_outbounds.text_y[0:100].apply(lambda txt: clean_text(txt))

# Save data ---------------------
with open('./data/x_text.pickle', 'wb') as handle:
    pickle.dump(x_text, handle)

with open('./data/y_text.pickle', 'wb') as handle:
    pickle.dump(y_text, handle)