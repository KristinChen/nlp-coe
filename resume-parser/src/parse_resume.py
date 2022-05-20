# -*- coding: utf-8 -*-

import pdfminer
from pdfminer.high_level import extract_text
import glob
import nltk
from string import punctuation
from nltk.corpus import stopwords
import re
from nltk.tag import pos_tag
from names_dataset import NameDataset
nd = NameDataset()
import spacy #numpy needs to be <1.20
nlp = spacy.load("en_core_web_sm")
import os
import pdfplumber
import pandas as pd

def read_file(path):
    txt = extract_text(path, codec='utf-8')
    return txt.encode('utf-8').decode('utf-8')

def read_file2(path):
    txt_lst = []
    with pdfplumber.open(path) as pdf:
        for p in range(len(pdf.pages)):
            txt_lst.append(pdf.pages[p].extract_text())
        txt = ' '.join(txt_lst)
    return txt.encode('utf-8').decode('utf-8')

def preprocess_text(txt):
    # verify type of input
    if not isinstance(txt, str):
        raise TypeError('Work with String Only')
    # verify len(text)
    if len(txt) < 1:
        raise ValueError('Work with nonempty String Only')
    #tokenize
    words = nltk.word_tokenize(txt)

    #remove stopwords and punctuations
    stopwords_set = set(stopwords.words('english'))
    simple_strings = [word for word in words if word not in stopwords_set if word not in punctuation]

    #remove unwanted characters
    res = [re.sub(r'[^\w\s]', '', word) for word in simple_strings]
    res1 = [re.sub(r'^abc(.*?)=[A-Z0-9]+(.*)', r'\1\2', word) for word in res]
    res2 = [str(res) for res in res1]

    res3 = [re.sub(r'/^[A-Za-z]+$/', '', res) for res in res2]
    res3 = [res.replace("ï", "i") if "ï" in res else res for res in res3]
    res3 = [re.sub('\d', '', res) for res in res3]
    res3 = [res.encode('ascii', "ignore").decode() for res in res3]

    return list(filter(None, res3)) #remove empty string from list #return tokens_txt

# name -------------------------------------------------------------------------
indian_last_names = ["Acharya", "Agarwal", "Khatri", "Ahuja", "Anand", "Laghari", "Patel",

                     "Reddy", "Bakshi", "Anthony", "Babu", "Arya", "Balakrishnan", "Banerjee", "Burman", "Bhatt",
                     "Basu", "Bedi", "Varma", "Dara", "Dalal", "Chowdhury",
                     "Chabra", "Chadha", "Chakrabarti", "Chawla", "Ahluwalia", "Amin", "Apte", "Datta", "Deol",
                     "Deshpande", "Dewan", "Lal", "Kohli", "Mangal", "Malhotra", "Jha",
                     "Joshi", "Kapadia", "Iyer", "Jain", "Khanna", "Grover", "Kaur", "Kashyap", "Gokhale", "Ghosh",
                     "Garg", "Dhar", "Gandhi", "Ganguly", "Gupta", "Das", "Chopra", "Dhawan",
                     "Dixit", "Dubey", "Haldar", "Kapoor", "Khurana", "Kulkarni", "Madan", "Bajwa", "Bhasin", "Chandra",
                     "Chauhan", "Deshmukh", "Dayal", "Dhillon", "Goswami", "Goel", "Mallick",
                     "Mahajan", "Kumar", "Mani", "Gill", "Mannan", "Biswas", "Batra", "Bawa", "Mehta", "Mukherjee",
                     "Saxena", "Zacharia", "Shah", "Ray", "Rao", "Purohit", "Parekh", "Thakur", "Singh", "Sharma",
                     "Seth", "Sachdev", "Ranganathan", "Puri", "Pandey", "Naidu", "Modi"]

chinese_last_names = ["Li", "Wang", "Zhang", "Liu", "Chen", "Yang", "Zhao", "Huang", "Zhou",

                      "Wu", "Xu", "Sun", "Hu", "Zhu", "Gao", "Lin", "He", "Guo", "Ma", "Luo", "Liang",

                      "Song", "Zheng", "Xie", "Han", "Tang", "Feng", "Yu", "Dong", "Xiao", "Cheng",

                      "Cao", "Yuan", "Deng", "Xu", "Fu", "Shen", "Zeng", "Peng", "Lu", "Su", "Lu", "Jiang", "Cai",
                      "Jia", "Ding", "Wei", "Xue", "Ye", "Yan",

                      "Yu", "Pan", "Du", "Dai", "Xia", "Zhong", "Wang", "Tian", "Ren", "Jiang", "Fan", "Fang", "Shi",
                      "Yao", "Tan", "Sheng", "Zou", "Xiong", "Jin", "Lu", "Hao", "Kong", "Bai", "Cui",

                      "Kang", "Mao", "Qio", "Qin", "Jiang", "Shu", "Shi", "Gu", "Hou", "Shao", "Meng", "Long", "Wan",
                      "Duan", "Zhang", "Qian", "Tang", "Yin", "Li", "Yi", "Chang", "Wu",

                      "Qiao", "He", "Lao", "Gong", "Wen"]

# chinese_last_names = [chinese_last_name.lower() for chinese_last_name in chinese_last_names]
# indian_last_names = [indian_last_name.lower() for indian_last_name in indian_last_names]

us_lnames = nd.get_top_names(n=2000, use_first_names = False, country_alpha2='US')
china_lnames = nd.get_top_names(n=2000, use_first_names = False, country_alpha2='CN')
indian_lnames = nd.get_top_names(n=2000, use_first_names = False, country_alpha2='IN')

ref_names = list(set(us_lnames['US'] + china_lnames['CN'] + chinese_last_names+ indian_lnames['IN']  + indian_last_names))

def get_name(sentence_txt, tokens_txt): #-> list of person names
    temp_person = []
    nltk_tagged = pos_tag(tokens_txt[:11])
    for k in range(10):
        if nltk_tagged[k][1] == 'NNP' and nltk_tagged[k + 1][1] == 'NNP':
            nltk_name = nltk_tagged[k][0] + ' ' + nltk_tagged[k + 1][0]
            temp_person.append(nltk_name)
            if len(temp_person) > 0:# Comment: stop as long as you got the nltk_name
                return temp_person
        if k > 1:
            if (tokens_txt[k].lower() in indian_last_names) or (tokens_txt[k].lower() in chinese_last_names):
                j = k - 1
                return([tokens_txt[j] + " " + tokens_txt[k]]) #Comment: what if kristin (jiating) chen?

    tagged = nlp(sentence_txt[0:200]) #Comment: assume names exist in the top section
    for word in tagged.ents:
        if word.label_ == "PERSON":
            temp_person.append(word.text)
        return temp_person


def get_name_from_ner(sentence_txt):
    temp_person = []
    tagged = nlp(sentence_txt[0:200]) #Comment: assume names exist in the top section
    for word in tagged.ents:
        if word.label_ == "PERSON":
            temp_person.append(word.text)
        return temp_person
    return list(set(temp_person))

def get_name_from_pos(tokens_txt):
    temp_person = []
    nltk_tagged = pos_tag(tokens_txt[:11])
    for k in range(10):
        if nltk_tagged[k][1] == 'NNP' and nltk_tagged[k + 1][1] == 'NNP':
            nltk_name = nltk_tagged[k][0] + ' ' + nltk_tagged[k + 1][0]
            temp_person.append(nltk_name)
            # if len(temp_person) > 0:  # Comment: stop as long as you got the nltk_name
            #     return temp_person
    return list(set(temp_person))

def get_name_from_ref(tokens_txt):
    temp_person = []
    for k in range(10):
        if k >= 1:
            j = k - 1
            if tokens_txt[k].capitalize() in ref_names:
                if len(tokens_txt[j]) < 2 and j != 0:
                    temp_person.append(tokens_txt[j-1] + " " + tokens_txt[j] + " " + tokens_txt[k])
                else:
                    temp_person.append(tokens_txt[j] + " " + tokens_txt[k])
    return(list(set(temp_person)))



# -----------------------------------------------------------------
RESERVED_WORDS = [
    'school',
    'college',
    'univers',
    'academy',
    'faculty',
    'institute',
    'faculdades',
    'Schola',
    'schule',
    'lise',
    'lyceum',
    'lycee',
    'polytechnic',
    'kolej',
    'ünivers',
    'okul',
    'University'
]

def extract_education(txt): # Comment: this is not solid
    edu=set()
    p = re.compile('(EDUCATION)?\n?(.*?),\s+(.*?),(.*?)')
    for m in re.finditer(p, txt):

        for word in RESERVED_WORDS:
            if word in m.group(2).lower():
                edu.add(m.group(2))
    return edu

# -----------------------------
def extract_phone_number(txt):
    PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    phone = re.findall(PHONE_REG, txt)

    if phone:
        number = ''.join(phone[0])

        if txt.find(number) >= 0 and len(number) < 16:
            return number
    return None

# ---------------------------- -
def extract_email(txt):
    EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
    return re.findall(EMAIL_REG, txt)

# -----------------------------------
def extract_phd_degree(txt):
    phd_set = set()
    p = re.compile('(Ph\.D).*,|(Ph\.D).*\\n')
    for m in re.finditer(p, txt):
        candidate_txt = m.group().split(',')[0]
        phd_set.add(' '.join(candidate_txt.split()))
    return phd_set



# ------------------------------------
def extract_master_degree(txt):
    masters_set = set()
    p = re.compile('(Master|MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech).*,|(Master|MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech).*\\n')  # greedy
    for m in re.finditer(p, txt):
        candidate_txt = m.group().split(',')[0] #Comment: to avoid getting 'MS Excel MS word'....
        if 'Master' in candidate_txt:
            return ' '.join(candidate_txt.split()) #immediate return master
        else:
            masters_set.add(' '.join(candidate_txt.split()))
    return masters_set



# ---------------------------------
def extract_bachelor_degree(txt):
    bachelors_set = set()
    p = re.compile('(Bachelor|BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech).*,|(Bachelor|BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech).*\\n')  # greedy
    for m in re.finditer(p, txt):
        candidate_txt = m.group().split(',')[0] #MS Excel MS word....
        if 'Bachelor' in candidate_txt:
            return ' '.join(candidate_txt.split()) #immediate return master
        else:
            bachelors_set.add(' '.join(candidate_txt.split()))
    return bachelors_set



# --------- aggregate -----------------------------
def get_info(txt):
    tokens_txt = preprocess_text(txt)
    name = get_name(txt, tokens_txt)
    name_ner = get_name_from_ner(txt)
    name_pos = get_name_from_pos(tokens_txt)
    name_ref = get_name_from_ref(tokens_txt)
    # education = extract_education(txt)
    phone_number = extract_phone_number(txt)
    email = extract_email(txt)
    master = extract_master_degree(txt)
    bachelor = extract_bachelor_degree(txt)
    phd = extract_phd_degree(txt)
    return {
        'name' : name,
        'name_ner': name_ner,
        'name_pos': name_pos,
        'name_ref': name_ref,
        'phone_number' : phone_number,
        'email' : email,
        'master_degree' : master,
        'bachelor_degree': bachelor,
        'phd_degree' : phd
    }

def parse_resume(path):
    index = os.path.basename(path)
    print(index)
    try:
        raw_txt = read_file(path)
    except pdfminer.pdfparser.PDFSyntaxError:
        return_dict = {'index': index,
                   'name': None,
                   'name_ner': None,
                   'name_pos': None,
                   'name_ref': None,
                   'phone_number': None,
                   'email': None,
                   'master_degree': None,
                   'bachelor_degree': None,
                   'phd_degree': None}
        return return_dict

    if not isinstance(raw_txt, str):
        return_dict = {   'index' : index,
                          'name': None,
                          'name_ner': None,
                          'name_pos': None,
                          'name_ref': None,
                          'phone_number': None,
                          'email': None,
                          'master_degree': None,
                          'bachelor_degree': None,
                          'phd_degree': None}
        return return_dict
    elif len(raw_txt) < 8:
        return_dict = {   'index' : index,
                          'name': None,
                          'name_ner': None,
                          'name_pos': None,
                          'name_ref': None,
                          'phone_number': None,
                          'email': None,
                          'master_degree': None,
                          'bachelor_degree': None,
                          'phd_degree': None}
    else:
        info_dict = get_info(raw_txt)
        return_dict = {**{'index': index}, **info_dict}
        return return_dict

# -------------- execution ------------------------------------------------------
folder = glob.glob(r"./data/resume_samples/*")
# file_path = folder[2]
#
# sample_txt = read_file(file_path).decode('utf-8')
# sample_txt = read_file2(file_path)
# tokens_txt = preprocess_text(sample_txt)
#
#
#
# get_name(sentence_txt=sample_txt, tokens_txt = tokens_txt)
# get_name_from_ner(sample_txt)
# get_name_from_pos(tokens_txt)
# get_name_from_ref(tokens_txt)
#
# extract_education(sample_txt[0:1000])
# doc = nlp(sample_txt)
# extract_phd_degree(sample_txt)
# doc = nlp(sample_txt)
# extract_master_degree(sample_txt)
# doc = nlp(sample_txt)
# extract_bachelor_degree(sample_txt)

df = pd.DataFrame()
for f in folder:
    return_dict = parse_resume(f)
    df = df.append(return_dict, ignore_index = True)

df.to_csv('matched_resumes.csv', index=False)

