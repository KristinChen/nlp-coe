# -*- coding: utf-8 -*-

from dateutil import parser
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
import datefinder
from datetime import datetime
from thefuzz import fuzz
# from pyresparser import ResumeParser
import numpy as np

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
            if tokens_txt[k].lower() in ref_names: #fuzzy matching
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
    p = re.compile('(Ph\.D|PhD|PhD\.).*,|(Ph\.D|PhD|PhD\.).*\\n')
    for m in re.finditer(p, txt):
        candidate_txt = m.group().split(',')[0]
        phd_set.add(' '.join(candidate_txt.split()))
    return phd_set

def extract_phd_degree_from_lines(txt):
    lines_txt = [l.strip() for l in txt.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    phd_degrees_lst = []
    for index, line in enumerate(lines_txt):
        if re.match('(Ph\.D|PhD|PhD\.)' ,line):
            phd_degrees_lst.append(line.strip())
    return [' '.join(m.split()) for m in phd_degrees_lst]

def extract_phd_degree_from_lines(txt):
    replace_words = ["|", "/", "."]

    lines_txt = [l.strip() for l in txt.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    phd_degrees_lst = []
    for index, line in enumerate(lines_txt):
        if re.match('(Ph\.D|PhD|PhD\.)' ,line):
            phd_degrees_lst.append(line.strip())
    if len(phd_degrees_lst)>0:
        phd_deg = [' '.join(m.split()) for m in phd_degrees_lst][0]
        phd_deg1 = re.sub("\d+|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", " ",phd_deg)
        phd_deg2 = re.sub("Current|Cumulative|GPA"," ",phd_deg1)
        for punct in replace_words:
            if punct in phd_deg2:
                phd_deg2 = phd_deg2.replace(punct," ")
    else:
        phd_deg2 = None
    return phd_deg2
# ------------------------------------
neglect_words = [
    'MSSQL',
    'MS LAB',
    'MSMQ',
    'MS-DB',
    'MS-Excel',
    'MS-Access',
    'MATLAB',
    'MARY',
    'MACHINE',
    'MAPE',
    'MSE',
    'MARKOV',
    'MAXIMIZATION',
    'MATPLOTLIB',
    'MATPLOT',
    'MATPLOTLIB',
    'MATRICES',
    'MATRIX',
    'MASSACHUSETTS',
    'Mssql',
    'Ms Lab',
    'Msmq',
    'Ms-db',
    'Ms-excel',
    'Ms-access',
    'Matlab',
    'Mary',
    'Machine',
    'Mape',
    'Mse',
    'Markov',
    'Maximization',
    'Matplotlib',
    'Matplot',
    'Matplotlib',
    'Matrices',
    'Matrix',
    'Massachusetts'
    'mssql',
    'ms lab',
    'msmq',
    'ms-db',
    'ms-excel',
    'ms-access',
    'matlab',
    'mary',
    'machine',
    'mape',
    'mse',
    'markov',
    'maximization',
    'matplotlib',
    'matplot',
    'matplotlib',
    'matrices',
    'matrix',
    'massachusetts'
]

def extract_master_degree(txt):
    ms_set = set()
    masters_set = set()
    p = re.compile('((?i:Master)|MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech).*,|((?i:Master)|MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech).*\\n')  # greedy
    for m in re.finditer(p, txt):
        candidate_txt = m.group().split(',')[0]
        for w in candidate_txt.split():
            if w in neglect_words:
                continue
            else:
                ms_set.add(' '.join(candidate_txt.split()))
        if re.search('(?i:Master)', candidate_txt):
            print(candidate_txt)
            masters_set.add(' '.join(candidate_txt.split()))
    if len(masters_set) > 0:
        return masters_set
    else:
        return ms_set

def is_valid_date(txt):
    return_lst = []
    matches = datefinder.find_dates(txt)
    for match in matches:
        #print(match)
        return_lst.append(match)
    if len(return_lst):
        return True
    else:
        return False


# replace_words = ["|","/","."]
# sample_txt = "Master of Science Information Systems | 3.54/4 Cumulative Current GPA January 2021"
# sample_txt = "Master of Professional Studies in Data Science, GPA: 3.74"
# sample_txt = "Master of Science in Business Analytics, Focus Area: Data Science GPA: 4.0"
# sample_txt = "Bachelor of Technology in Chemical Engineering; Cumulative GPA: 8.57/10.0"
# sample_txt = "'Bachelor of Science in Industrial Engineering, summa cum laude (Equiv. GPA: 3.9) June 2011 â€“ June 2016'"
# sample_txt1 = re.sub("\d+|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", " ",sample_txt)
# sample_txt2 = re.sub("Current|Cumulative|GPA"," ",sample_txt1)
# #sample_txt3 = re.sub(r'[\w\s]'," ",sample_txt2)
# for punct in replace_words:
#     if punct in sample_txt2:
#         sample_txt2 = sample_txt2.replace(punct," ")
#
# re.sub(r'\|.*Equiv\.|Current|Cumulative|GPA.*',"",sample_txt)


def extract_master_degree_from_lines(txt):
    year ="(\d{4})"
    month = "((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))"
    lines_txt = [l.strip() for l in txt.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    master_degrees_lst = []
    ms_lst = []
    for index, line in enumerate(lines_txt):
        if re.match('(?i:Master)' ,line):
            master_degrees_lst.append(line.strip())
        if re.match('MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech' ,line):
            ms_lst.append(line.strip())
    if len(master_degrees_lst) > 0:
        # cleaned_master_degress_lst = []
        # for d in master_degrees_lst:
        #     cleaned_master_degress_lst.append(' '.join([m for m in d.split() if m != month or not m.isdigit()]))
        return [' '.join(m.split()) for m in master_degrees_lst]
    else:
        return [' '.join(m.split()) for m in ms_lst]

def extract_master_degree_from_lines(txt):
    year ="(\d{4})"
    month = "((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))"
    replace_words = ["|", "/", "."]

    lines_txt = [l.strip() for l in txt.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    master_degrees_lst = []
    ms_lst = []
    for index, line in enumerate(lines_txt):
        if re.match('(?i:Master)' ,line):
            master_degrees_lst.append(line.strip())
        if re.match('MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech' ,line):
            ms_lst.append(line.strip())
    if len(master_degrees_lst) > 0:
        # cleaned_master_degress_lst = []
        # for d in master_degrees_lst:
        #     cleaned_master_degress_lst.append(' '.join([m for m in d.split() if m != month or not m.isdigit()]))
        mast_deg = [' '.join(m.split()) for m in master_degrees_lst][0]
        mast_deg1 = re.sub("\d+|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", " ",mast_deg)
        mast_deg2 = re.sub("Current|Cumulative|GPA"," ",mast_deg1)
        for punct in replace_words:
            if punct in mast_deg2:
                mast_deg2 = mast_deg2.replace(punct," ")
        return mast_deg2
    elif len(ms_lst) > 0:
        mast_deg = [' '.join(m.split()) for m in ms_lst][0]
        mast_deg1 = re.sub("\d+|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", " ",mast_deg)
        mast_deg2 = re.sub("Current|Cumulative|GPA"," ",mast_deg1)
        for punct in replace_words:
            if punct in mast_deg2:
                mast_deg2 = mast_deg2.replace(punct," ")
        print(mast_deg2)
        return mast_deg2
    else:
        mast_deg2 = None
        return mast_deg2
# ---------------------------------
# doesn't match: \nBachelors of Management Studies (concentration in marketing)
def extract_bachelor_degree(txt):
    bachelors_set = set()
    ba_set = set()
    p = re.compile('((?i:Bachelor)|BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech).*,|((?i:Bachelor)|BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech).*\\n')  # greedy
    for m in re.finditer(p, txt):
        candidate_txt = m.group().split(',')[0] #MS Excel MS word...
        print(candidate_txt)
        ba_set.add(' '.join(candidate_txt.split()))
        if re.search('(?i:Bachelor)', candidate_txt):
            bachelors_set.add(' '.join(candidate_txt.split()))
        if len(bachelors_set) > 0:
            return bachelors_set
        else:
            return ba_set

def extract_bachelor_degree_from_lines(txt):
    lines_txt = [l.strip() for l in txt.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    bachelors_set = []
    ba_set = []
    for index, line in enumerate(lines_txt):
        if re.match('(?i:Bachelor)' ,line):
            bachelors_set.append(line.strip())
        if re.match('BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech' ,line):
            ba_set.append(line.strip())

    if len(bachelors_set) > 0:
        return [' '.join(m.split()) for m in bachelors_set]
    else:
        return [' '.join(m.split()) for m in ba_set]

def extract_bachelor_degree_from_lines(txt):
    replace_words = ["|", "/", "."]

    lines_txt = [l.strip() for l in txt.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    bachelors_set = []
    ba_set = []
    for index, line in enumerate(lines_txt):
        if re.match('(?i:Bachelor)' ,line):
            bachelors_set.append(line.strip())
        if re.match('BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech' ,line):
            ba_set.append(line.strip())

    if len(bachelors_set) > 0:
        bach_deg = [' '.join(m.split()) for m in bachelors_set][0]
        bach_deg1 = re.sub("\d+|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", " ",bach_deg)
        bach_deg2 = re.sub("Current|Cumulative|GPA"," ",bach_deg1)
        for punct in replace_words:
            if punct in bach_deg2:
                bach_deg2 = bach_deg2.replace(punct," ")
        print(bach_deg2)
        return bach_deg2
    elif len(ba_set) > 0:
        bach_deg = [' '.join(m.split()) for m in ba_set][0]
        bach_deg1 = re.sub("\d+|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Sept|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)", " ",bach_deg)
        bach_deg2 = re.sub("Current|Cumulative|GPA"," ",bach_deg1)
        for punct in replace_words:
            if punct in bach_deg2:
                bach_deg2 = bach_deg2.replace(punct," ")
        print(bach_deg2)
        return bach_deg2
    else:
        bach_deg2 = None
        return bach_deg2
# -----------------------------------------------------
def extract_phd_school(txt):
    p = re.compile(
        '(?i)(school|college|univers|academy|faculty|institute|faculdades|Schola|schule|lise|lyceum|lycee|polytechnic|kolej|ünivers|okul|University)(.|\n)*(Ph.D.|PhD|PhD.)')
    for m in re.finditer(p, txt):
        return m.group(0).split("\n")[0]

def extract_ms_school(txt):
    p = re.compile(
        '(?i)(school|college|univers|academy|faculty|institute|faculdades|Schola|schule|lise|lyceum|lycee|polytechnic|kolej|ünivers|okul|University)(.|\n)*(M.S.|M.S|M.Sc.|MS|Masters|(?i:Master))')
    for m in re.finditer(p, txt):
        return m.group(0).split("\n")[0]


## Using regex for finding university nearest to Bachelor's degree
def extract_bach_school(txt):
    p = re.compile(
        '(?i)(school|college|academy|faculty|institute|faculdades|Schola|schule|lise|lyceum|lycee|polytechnic|kolej|ünivers|okul|University)(?:(?!(?i)(school|college|academy|faculty|institute|faculdades|Schola|schule|lise|lyceum|lycee|polytechnic|kolej|ünivers|okul|University)|(Bachelors|B.tech|Bachelor\'s|B.A.|BS|B.Sc.|B.S|B.S.))[\s\S])*(Bachelors|B.tech|Bachelor\'s|B.A.|BS|B.Sc.|B.S|B.S.)')
    for m in re.finditer(p, txt):
        return m.group(0).split("\n")[0]

# date -------------------------------------------------------------------------


def extract_master_date(txt1):
    lines_txt = [l.strip() for l in txt1.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]

    for index, line in enumerate(lines_txt):
            if re.match('(?i:Master)' ,line):
                masters_date_lst = []
                start_idx = index
                for i, l in enumerate(lines_txt[start_idx:start_idx + 8]):
                    print(i, l)
                    if datefinder.find_dates(l):
                        matches = datefinder.find_dates(l)
                        for match in matches:
                            print(match)
                            masters_date_lst.append(match)
                        if len(masters_date_lst) > 0:
                            return [d.strftime("%m/%Y") for d in masters_date_lst]
            elif re.match('MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech' ,line):
                return_set = []
                start_idx = index
                for _, l in enumerate(lines_txt[start_idx:start_idx + 8]):
                    if datefinder.find_dates(l):
                        matches = datefinder.find_dates(l)
                        for match in matches:
                            print(match)
                            return_set.append(match)
                        return [d.strftime("%m/%Y") for d in return_set]

def extract_bachelor_date(txt1):
    lines_txt = [l.strip() for l in txt1.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]

    for index, line in enumerate(lines_txt):
        if re.match('(?i:Bachelor)', line):
            bachelor_date_lst = []
            start_idx = index
            for i, l in enumerate(lines_txt[start_idx:start_idx + 8]):
                print(i, l)
                if datefinder.find_dates(l):
                    matches = datefinder.find_dates(l)
                    for match in matches:
                        print(match)
                        bachelor_date_lst.append(match)
                    if len(bachelor_date_lst) > 0:
                        return [d.strftime("%m/%Y") for d in bachelor_date_lst]
        elif re.match('BS|B\.S\.|BA|B\.A\.|B\.S\.E|B\.tech' ,line):
            return_set = []
            start_idx = index
            for _, l in enumerate(lines_txt[start_idx:start_idx + 8]):
                if datefinder.find_dates(l):
                    matches = datefinder.find_dates(l)
                    for match in matches:
                        print(match)
                        return_set.append(match)
                    return [d.strftime("%m/%Y") for d in return_set]

def extract_phd_date(txt1):
    lines_txt = [l.strip() for l in txt1.split('\n') if len(l.strip()) > 0 and len(l.strip().split()) > 0]
    phd_date_lst = []
    for index, line in enumerate(lines_txt):
        if re.match('(Ph\.D|PhD|PhD\.)' ,line):
            start_idx = index
            for i, l in enumerate(lines_txt[start_idx:start_idx + 8]):
                print(i, l)
                if datefinder.find_dates(l):
                    matches = datefinder.find_dates(l)
                    for match in matches:
                        print(match)
                        phd_date_lst.append(match)
                    if len(phd_date_lst) > 0:
                        return [d.strftime("%m/%Y") for d in phd_date_lst]

    return phd_date_lst



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
    master_degree = extract_master_degree(txt)
    bachelor_degree = extract_bachelor_degree(txt)
    phd_degree = extract_phd_degree(txt)
    master_degree2 = extract_master_degree_from_lines(txt)
    bachelor_degree2 = extract_bachelor_degree_from_lines(txt)
    phd_degree2 = extract_phd_degree_from_lines(txt)

    master_school = extract_ms_school(txt)
    bachelor_school = extract_bach_school(txt)
    phd_school = extract_phd_school(txt)
    master_graduation_date = extract_master_date(txt)
    bachelor_graduation_date = extract_bachelor_date(txt)
    phd_graduation_date = extract_phd_date(txt)
    return {
        'name' : name,
        'name_ner': name_ner,
        'name_pos': name_pos,
        'name_ref': name_ref,
        'phone_number' : phone_number,
        'email' : email,
        'master_degree' : master_degree,
        'bachelor_degree': bachelor_degree,
        'phd_degree' : phd_degree,
        'master_degree2': master_degree2,
        'bachelor_degree2': bachelor_degree2,
        'phd_degree2': phd_degree2,
        'master_school': master_school,
        'bachelor_school': bachelor_school,
        'phd_school': phd_school,
        'master_graduation_date': master_graduation_date,
        'bachelor_graduation_date': bachelor_graduation_date,
        'phd_graduation_date': phd_graduation_date
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
                   'phd_degree': None,
                   'master_degree2': None,
                   'bachelor_degree2': None,
                   'phd_degree2': None,
                   'master_school': None,
                   'bachelor_school': None,
                   'phd_school': None,
                   'master_graduation_date': None,
                   'bachelor_graduation_date': None,
                   'phd_graduation_date': None
                       }
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
                          'phd_degree': None,
                          'master_degree2': None,
                          'bachelor_degree2': None,
                          'phd_degree2': None,
                          'master_school': None,
                          'bachelor_school': None,
                          'phd_school': None,
                          'master_graduation_date': None,
                          'bachelor_graduation_date': None,
                          'phd_graduation_date': None}
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
                          'phd_degree': None,
                          'master_degree2': None,
                          'bachelor_degree2': None,
                          'phd_degree2': None,
                          'master_school': None,
                          'bachelor_school': None,
                          'phd_school': None,
                          'master_graduation_date': None,
                          'bachelor_graduation_date': None,
                          'phd_graduation_date': None}
        return return_dict
    else:
        info_dict = get_info(raw_txt)
        return_dict = {**{'index': index}, **info_dict}
        return return_dict

# -------------- execution ------------------------------------------------------
folder = glob.glob(r"./data/resume_samples/*")
file_path = folder[52]
sample_txt = read_file(file_path)
# sample_txt2 = read_file2(file_path)
tokens_txt = preprocess_text(sample_txt)
#
get_name(sentence_txt=sample_txt, tokens_txt = tokens_txt)
get_name_from_ner(sample_txt)
get_name_from_pos(tokens_txt)
get_name_from_ref(tokens_txt)
extract_education(sample_txt[0:1000])
# doc = nlp(sample_txt)
extract_phd_degree(sample_txt)
extract_phd_degree_from_lines(sample_txt)
extract_phd_school(sample_txt)
extract_phd_date(sample_txt)
extract_master_degree(sample_txt)
extract_master_degree_from_lines(sample_txt)
extract_ms_school(sample_txt)
extract_master_date(sample_txt)
extract_bachelor_degree(sample_txt)
extract_bachelor_degree_from_lines(sample_txt)
extract_bach_school(sample_txt)
extract_bachelor_date(sample_txt)

df = pd.DataFrame()
for f in folder:
    return_dict = parse_resume(f)
    df = df.append(return_dict, ignore_index = True)

df.to_csv('matched_resumes.csv', index=False)

df.shape #201-182?
# # --------- performance ------------------------------
df = pd.read_csv('./matched_resumes.csv', index_col=False)
gt_df = pd.read_csv('./Data/GroundTruth.csv', index_col=False)
compared_df = pd.merge(gt_df, df, how = 'left', on = 'index')

gt_df.shape #182
# gt_df.columns
# gt_df.master_graduation_date
gt_df.columns
df.columns
df.shape
compared_df.columns
compared_df.shape

def unlist(x):
    if isinstance(x, list) and len(x) != 0:
        for n in x:
            return n
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    else:
        return x

# -------------------- NAME -----------------------------------------------------------
compared_df['full_name'] = compared_df['full_name'].fillna('')
compared_df['full_name'].replace(to_replace=[None], value='', inplace=True)
compared_df['full_name'] = compared_df.full_name.apply(lambda x: str(x).capitalize())
compared_df['full_name'] = compared_df['full_name'].apply(lambda x:x.lower())

compared_df['unlisted_name'] = compared_df.name.apply(lambda x: str(x).replace("['", '').replace("']", '').replace("[", '').replace("]", ''))
compared_df['unlisted_name'] = compared_df['unlisted_name'].fillna('')
compared_df['unlisted_name'].replace(to_replace=[None], value='', inplace=True)
compared_df['unlisted_name'] = compared_df['unlisted_name'].apply(lambda x:x.lower())

compared_df['unlisted_name_ner'] = compared_df.name_ner.apply(lambda x: str(x).replace("['", '').replace("']", '').replace("[", '').replace("]", '').replace("\\n", ''))
compared_df['unlisted_name_ner'] = compared_df['unlisted_name_ner'].fillna('')
compared_df['unlisted_name_ner'].replace(to_replace=[None], value='', inplace=True)
compared_df['unlisted_name_ner'] = compared_df['unlisted_name_ner'].apply(lambda x:x.lower())

compared_df['unlisted_name_pos'] = compared_df.name_pos.apply(lambda x: str(x).replace("['", '').replace("']", '').replace("[", '').replace("]", '').replace("'", '').replace("'", ''))
compared_df['unlisted_name_pos'] = compared_df['unlisted_name_pos'].fillna('')
compared_df['unlisted_name_pos'].replace(to_replace=[None], value='', inplace=True)
compared_df['unlisted_name_pos'] = compared_df['unlisted_name_pos'].apply(lambda x:x.lower())

compared_df['unlisted_name_ref'] = compared_df.name_ref.apply(lambda x: str(x).replace("['", '').replace("']", '').replace("[", '').replace("]", '').replace("'", '').replace("'", ''))
compared_df['unlisted_name_ref'] = compared_df['unlisted_name_ref'].fillna('')
compared_df['unlisted_name_ref'].replace(to_replace=[None], value='', inplace=True)
compared_df['unlisted_name_ref'] = compared_df['unlisted_name_ref'].apply(lambda x:x.lower())

comp_name1 = compared_df.apply(lambda x: fuzz.ratio(x['unlisted_name'], x['full_name']), axis = 1)
token_set_comp_name1 = compared_df.apply(lambda x: fuzz.token_set_ratio(x['unlisted_name'], x['full_name']), axis = 1)
token_sort_comp_name1 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['unlisted_name'], x['full_name']), axis = 1)

np.mean(comp_name1) #76.1208
np.mean(token_set_comp_name1) #79.6098
np.mean(token_sort_comp_name1) #75.8187

# comp_name2 = compared_df.apply(lambda x: fuzz.ratio(x['unlisted_name_ner'], x['full_name']), axis = 1)
# np.mean(comp_name2) #44.55223880597015
# comp_name3 = compared_df.apply(lambda x: fuzz.ratio(x['unlisted_name_pos'], x['full_name']), axis = 1)
# np.mean(comp_name3) #51.78606965174129
# comp_name4 = compared_df.apply(lambda x: fuzz.ratio(x['unlisted_name_ref'], x['full_name']), axis = 1)
# np.mean(comp_name4) #54.17910447761194

# -------------------- PHONE NUMBER -----------------------------------------------------
compared_df.columns
# phone_number, phone
from string import punctuation

def clean_phone_number(x):
    if isinstance(x, str):
        return_x = ''.join([p for p in x if p not in punctuation]).replace(" ", "")
        print(return_x)
        if len(return_x) != 10 and return_x.startswith('1'):
            return return_x[1:]
        else:
            return return_x
    if not isinstance(x, str):
        return ''

compared_df['number_pred'] = compared_df.phone_number.apply(lambda x:clean_phone_number(x))
compared_df['phone_ref'] = compared_df.phone.apply(lambda x:clean_phone_number(x))


comp_phone = compared_df.apply(lambda x: fuzz.ratio(x['number_pred'], x['phone_ref']), axis = 1)
np.mean(comp_phone) #81.4340

# --------------- EMAIL ----------------------
compared_df.columns #email_x #email_y
compared_df.email_x
compared_df.email_y

compared_df['email_pref'] = compared_df.email_x.apply(lambda x: str(x).replace("['", '').replace("']", '').replace("[", '').replace("]", ''))
compared_df['email_true'] = compared_df.email_y.apply(lambda x: str(x).replace("['", '').replace("']", '').replace("[", '').replace("]", ''))
comp_email = compared_df.apply(lambda x: fuzz.ratio(x['email_pref'], x['email_true']), axis = 1)
np.mean(comp_email) #98.5274

# school ------------------------------------------------
compared_df.columns

# add them together in a list
compared_df.master_school_x
compared_df.Master_school_2
compared_df.Master_school_3

compared_df.master_school_y #remove {}


def clean_school_set(set_x):
    if not isinstance(set_x, set):
        return ''
    else:
        x = unlist(list(set_x))
        print(x)
        if isinstance(x, str):
            return_x = ''.join([p for p in x if p not in punctuation])
            print(return_x)
            return re.sub(' +', ' ', return_x).lower()

def clean_school_string(string_x):

    if isinstance(string_x, str):
        # return_x = ''.join([p for p in string_x if p not in punctuation])
        # print(return_x)
        return re.sub(' +', ' ', string_x).lower()
    else:
        return ''

compared_df.master_school_x = compared_df.master_school_x.apply(lambda x: clean_school_string(x))
compared_df.Master_school_2 = compared_df.Master_school_2.apply(lambda x: clean_school_string(x))
compared_df.Master_school_3 = compared_df.Master_school_3.apply(lambda x: clean_school_string(x))

compared_df['cleaned_master_school_y'] = compared_df.master_school_y.apply(lambda x: str(x).replace("{'", '').replace("'}", '')).apply(lambda x: clean_school_string(x))

compared_df['all_masters_x'] = compared_df.master_school_x + ' ' +  compared_df.Master_school_2 + ' ' + compared_df.Master_school_3
compared_df['all_masters_x']= compared_df.all_masters_x.apply(lambda x: clean_school_string(x))

comp_master_school = compared_df.apply(lambda x: fuzz.ratio(x['cleaned_master_school_y'], x['all_masters_x']), axis = 1)
np.mean(comp_master_school) #51.5824
comp_master_school2 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['cleaned_master_school_y'], x['all_masters_x']), axis = 1)
np.mean(comp_master_school2) #55.2417
comp_master_school3 = compared_df.apply(lambda x: fuzz.token_set_ratio(x['cleaned_master_school_y'], x['all_masters_x']), axis = 1)
np.mean(comp_master_school) #51.5824
np.mean(comp_master_school2) #55.2417
np.mean(comp_master_school3) #70.2417


# add them together in a list
compared_df.bachelor_school_x
compared_df.Bach_school_2
compared_df.bachelor_school_y #remove {}


compared_df.bachelor_school_x = compared_df.bachelor_school_x.apply(lambda x: clean_school_string(x))
compared_df.bachelor_school_2 = compared_df.Bach_school_2.apply(lambda x: clean_school_string(x))
compared_df['all_bachelor_x'] = compared_df.bachelor_school_x + ' ' +  compared_df.bachelor_school_x
compared_df['all_bachelor_x']= compared_df.all_bachelor_x.apply(lambda x: clean_school_string(x))

compared_df['cleaned_bachelor_school_y'] = compared_df.bachelor_school_y.apply(lambda x: str(x).replace("{'", '').replace("'}", '')).apply(lambda x: clean_school_string(x))

comp_bachelor_school = compared_df.apply(lambda x: fuzz.ratio(x['all_bachelor_x'], x['cleaned_bachelor_school_y']), axis = 1)
comp_bachelor_school2 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['all_bachelor_x'], x['cleaned_bachelor_school_y']), axis = 1)
comp_bachelor_school3 = compared_df.apply(lambda x: fuzz.token_set_ratio(x['all_bachelor_x'], x['cleaned_bachelor_school_y']), axis = 1)

np.mean(comp_bachelor_school) #32.7362
np.mean(comp_bachelor_school2) #32.7362
np.mean(comp_bachelor_school3) #55.8461


compared_df.phd_school_x = compared_df.phd_school_x.apply(lambda x: clean_school_string(x))
compared_df['cleaned_phd_school_y'] = compared_df.phd_school_y.apply(lambda x: str(x).replace("nan", '').replace("'}", '')).apply(lambda x: clean_school_string(x))
comp_phd_school = compared_df.apply(lambda x: fuzz.token_set_ratio(x['cleaned_phd_school_y'], x['phd_school_x']), axis = 1)
comp_phd_school1 = compared_df.apply(lambda x: fuzz.ratio(x['cleaned_phd_school_y'], x['phd_school_x']), axis = 1)
comp_phd_school2 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['cleaned_phd_school_y'], x['phd_school_x']), axis = 1)
np.mean(comp_phd_school) #94.4450
np.mean(comp_phd_school1) #94.4560
np.mean(comp_phd_school2) #0.7032





# --------------------------------




# ---------------------- compare to other packages -------------------------------------------------------------------
from resume_parser import resumeparse
# data = resumeparse.read_file('/path/to/resume/file')
# data = ResumeParser(file_path).get_extracted_data()

# temp_person = []
# tagged = nlp(sentence_txt[0:200]) #Comment: assume names exist in the top section
#     for word in tagged.ents:
#         if word.label_ == "PERSON":
#             temp_person.append(word.text)

pyresparser_df = pd.read_csv('pyres_df.csv', index_col = False)
pyresparser_df.columns
pyresparser_df['degree']
pyresparser_df['college_name']
gt_df = pd.read_csv('./Data/GroundTruth.csv', index_col=False)
compared_df2 = pd.merge(gt_df, pyresparser_df, how = 'left', on = 'index')
compared_df2.loc[0, 'college_name'] #missing


compared_df2['degree'] = compared_df2.degree.apply(lambda x: clean_degree_string(x))
compared_df2['cleaned_master_degree'] = compared_df2.master_degree.apply(lambda x: clean_degree_string(x))
compared_df2['cleaned_master_degree2'] = compared_df2.Master_Degree_2.apply(lambda x: clean_degree_string(x))
compared_df2['cleaned_master_degree3'] = compared_df2.Master_Degree_3.apply(lambda x: clean_degree_string(x))
compared_df2['cleaned_bachelor_degree'] = compared_df2.bachelor_degree.apply(lambda x: clean_degree_string(x))
compared_df2['cleaned_Bach_Degree_2'] = compared_df2.Bach_Degree_2.apply(lambda x: clean_degree_string(x))
compared_df2['phd_degree'] = compared_df2.phd_degree.apply(lambda x: clean_degree_string(x))

compared_df2['all_degrees'] = compared_df2.cleaned_master_degree + ' ' + \
                               compared_df2.cleaned_master_degree2 + ' ' + \
                               compared_df2.cleaned_master_degree3 + ' ' + \
                               compared_df2.cleaned_bachelor_degree + ' ' + \
                               compared_df2.cleaned_Bach_Degree_2 + ' ' + \
                               compared_df2.phd_degree.apply(lambda x: clean_degree_string(x))

comp_degrees = compared_df2.apply(lambda x: fuzz.ratio(x['degree'], x['all_degrees']), axis = 1)
np.mean(comp_degrees) #26.5109
comp_degrees1 = compared_df2.apply(lambda x: fuzz.token_sort_ratio(x['degree'], x['all_degrees']), axis = 1)
np.mean(comp_degrees1) #29.1868
comp_degrees2 = compared_df2.apply(lambda x: fuzz.token_set_ratio(x['degree'], x['all_degrees']), axis = 1)
np.mean(comp_degrees2) #40.1153

masters_comp = pd.concat([
    compared_df['index'],
    compared_df.master_degree_x,
    compared_df.Master_Degree_2,
    compared_df.Master_Degree_3,
    compared_df.all_masters_x,
           compared_df.cleaned_master_degree_y,
           compared_df.cleaned_master_degree2,
           comp_master,
           comp_master21
           ], axis = 1)
masters_comp.to_csv('masters_comp.csv', index=False)

# date --------------------------------------------
compared_df[["phd_graduation_date_x",
             "phd_graduation_date_y",
             "master_graduation_date_x",
             "master_graduation_date_y","bachelor_graduation_date_x","bachelor_graduation_date_y"]]
def clean_degree_string(string_x):

    if isinstance(string_x, str):
        # return_x = ''.join([p for p in string_x if p not in punctuation])
        # print(return_x)
        return re.sub(' +', ' ', string_x).lower()
    else:
        return ''
def date_formatting(dt_string):
    dt_string = ''.join([i for i in dt_string if i not in punctuation])
    #dt_string = "Sep. 2021"
    # Convert string to datetime object
    try:
        dt_object = datetime.strptime(dt_string, "%m/%Y")
        return dt_object.strftime("%b %Y")
    except:
        try:
            dt_object = datetime.strptime(dt_string, "%b %Y")
            return dt_object.strftime("%b %Y")
        except:
            try:
                dt_object = datetime.strptime(dt_string, "%b. %Y")
                return dt_object.strftime("%b %Y")
            except:
                try:
                    dt_object = datetime.strptime(dt_string, "%b%Y")
                    return dt_object.strftime("%b %Y")
                except:
                    try:
                        dt_object = datetime.strptime(dt_string, "%b'%y")
                        return dt_object.strftime("%b %Y")
                    except:
                        try:
                            dt_object = datetime.strptime(dt_string, "%B %Y")
                            return dt_object.strftime("%b %Y")
                        except:
                            try:
                                dt_object = datetime.strptime(dt_string, "%B %Y")
                                return dt_object.strftime("%b %Y")
                            except:
                                try:
                                    dt_object = datetime.strptime(dt_string, "%m%Y")
                                    return dt_object.strftime("%b %Y")
                                except:
                                    pass

# -------- master date
compared_df["master_graduation_date_x1"] = compared_df["master_graduation_date_x"].apply(lambda x:date_formatting(str(x))).apply(lambda x: clean_degree_string(x))
compared_df["Master_graduation_date_x2"] = compared_df["Master_graduation_date_2"].apply(lambda x:date_formatting(str(x))).apply(lambda x: clean_degree_string(x))
compared_df["Master_graduation_date_x3"] = compared_df["Master_graduation_date_3"].apply(lambda x:date_formatting(str(x))).apply(lambda x: clean_degree_string(x))
compared_df['all_master_dates'] = compared_df.master_graduation_date_x1 + ' ' + \
                               compared_df.Master_graduation_date_x2 + ' ' + \
                               compared_df.Master_graduation_date_x3



compared_df["master_graduation_date_y1"] = compared_df["master_graduation_date_y"].apply(lambda x:date_formatting(str(x)[2:len(str(x))-2])).apply(lambda x: clean_degree_string(x))
comp_dates = compared_df.apply(lambda x: fuzz.ratio(x['all_master_dates'], x['master_graduation_date_y1']), axis = 1)
np.mean(comp_dates) #20.9945
comp_dates2 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['all_master_dates'], x['master_graduation_date_y1']), axis = 1)
np.mean(comp_dates2) #43.7582
comp_dates2 = compared_df.apply(lambda x: fuzz.token_set_ratio(x['all_master_dates'], x['master_graduation_date_y1']), axis = 1)
np.mean(comp_dates2) #23.0164

# bachelor date ---------------
compared_df["bachelor_graduation_date_x1"] = compared_df["bachelor_graduation_date_x"].apply(lambda x:date_formatting(str(x))).apply(lambda x: clean_degree_string(x))
compared_df["bachelor_graduation_date_x2"] = compared_df["Bach_graduation_date_2"].apply(lambda x:date_formatting(str(x))).apply(lambda x: clean_degree_string(x))
compared_df['all_bachelor_dates'] = compared_df.bachelor_graduation_date_x1 + ' ' + \
                               compared_df.bachelor_graduation_date_x2

compared_df["bachelor_graduation_date_y1"] = compared_df["bachelor_graduation_date_y"].apply(lambda x:date_formatting(str(x)[2:len(str(x))-2])).apply(lambda x: clean_degree_string(x))
comp_dates = compared_df.apply(lambda x: fuzz.ratio(x['all_bachelor_dates'], x['bachelor_graduation_date_y1']), axis = 1)
np.mean(comp_dates) #21.4175
comp_dates2 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['all_bachelor_dates'], x['bachelor_graduation_date_y1']), axis = 1)
np.mean(comp_dates2) #39.0329
comp_dates2 = compared_df.apply(lambda x: fuzz.token_set_ratio(x['all_bachelor_dates'], x['bachelor_graduation_date_y1']), axis = 1)
np.mean(comp_dates2) #22.0549

# phd date ---------------
compared_df["phd_graduation_date_x1"] = compared_df["phd_graduation_date_x"].apply(lambda x:date_formatting(str(x))).apply(lambda x: clean_degree_string(x))
compared_df["phd_graduation_date_y1"] = compared_df["phd_graduation_date_y"].apply(lambda x:date_formatting(str(x)[2:len(str(x))-2])).apply(lambda x: clean_degree_string(x))
comp_dates = compared_df.apply(lambda x: fuzz.ratio(x['phd_graduation_date_x1'], x['phd_graduation_date_y1']), axis = 1)
np.mean(comp_dates) #99.4505
comp_dates2 = compared_df.apply(lambda x: fuzz.token_sort_ratio(x['phd_graduation_date_x1'], x['phd_graduation_date_y1']), axis = 1)
np.mean(comp_dates2) #99.4505
comp_dates2 = compared_df.apply(lambda x: fuzz.token_set_ratio(x['phd_graduation_date_x1'], x['phd_graduation_date_y1']), axis = 1)
np.mean(comp_dates2) #0
