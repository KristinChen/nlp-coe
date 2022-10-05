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

# def read_file (in: path, out: text)
# def preprocess_text (in: ordinary_text, out: cleaned_text)
# def get_info (in: text, out: {resume_idx1:{name: 'kristin chen'}, {email: 'helloworld@gamil.com'}})
## def: get_name (in: text, out: {name: }
## def: get_degree_info (in: text: out :{degree1: {name: degree_name, level: phd, graduation_date: 01/02/2022}}
## def: get_email (in: text, out: {email: }}
## def: get_phone_number

folder = glob.glob(r"./nlp-coe/resume-parser/data/resume_samples/*")
file_path = folder[1]

def read_file(path):
    # with pdfplumber.open(path) as pdf:
    #     first_page = pdf.pages[0] # Question: why first page?
    #     txt1 = first_page.extract_text()

    txt = extract_text(path, codec='utf-8')
    return txt

sample_txt = read_file(file_path)

'''need to be happened every steps'''
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

tokens_txt = preprocess_text(sample_txt)
# -------------------------------------------------------------------------
us_names = nd.get_top_names(n=1000, gender='Male', country_alpha2='US')['US']['M']
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

chinese_last_names = [chinese_last_name.lower() for chinese_last_name in chinese_last_names]
indian_last_names = [indian_last_name.lower() for indian_last_name in indian_last_names]

def get_name(sentence_txt, tokens_txt): #-> list of person names
    temp_person = []
    nltk_tagged = pos_tag(tokens_txt[:11])
    for k in range(10):
        if nltk_tagged[k][1] == 'NNP' and nltk_tagged[k + 1][1] == 'NNP':
            nltk_name = nltk_tagged[k][0] + ' ' + nltk_tagged[k + 1][0]
            temp_person.append(nltk_name)
            if len(temp_person) > 0:# Comment: stop as long as you got the nltk_name
                return temp_person

        # for chunk in doc.noun_chunks:
        #     print(chunk.text) #Comment: maybe try noun chunks to get the name

        if k > 1:
            if (tokens_txt[k].lower() in indian_last_names) or (tokens_txt[k].lower() in chinese_last_names):
                j = k - 1
                return([tokens_txt[j] + " " + tokens_txt[k]]) #Comment: what if kristin (jiating) chen?

    tagged = nlp(sentence_txt[0:200]) #Comment: assume names exist in the top section
    for word in tagged.ents:
        if word.label_ == "PERSON":
            temp_person.append(word.text)
        return temp_person

get_name(sentence_txt=sample_txt, tokens_txt = tokens_txt)

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
extract_education(sample_txt[0:1000])

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

doc = nlp(sample_txt)
extract_phd_degree(sample_txt)

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

doc = nlp(sample_txt)
extract_master_degree(sample_txt)

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

doc = nlp(sample_txt)
extract_bachelor_degree(sample_txt)

# --------- aggregate -----------------------------
def get_info(txt):
    tokens_txt = preprocess_text(txt)
    name = get_name(txt, tokens_txt)
    education = extract_education(txt)
    phone_number = extract_phone_number(txt)
    email = extract_email(txt)
    master = extract_master_degree(txt)
    bachelor = extract_bachelor_degree(txt)
    phd = extract_phd_degree(txt)
    return {
        'name' : name,
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
                   'phone_number': None,
                   'email': None,
                   'master_degree': None,
                   'bachelor_degree': None,
                   'phd_degree': None}
        return return_dict

    if not isinstance(raw_txt, str):
        return_dict = {   'index' : index,
                          'name': None,
                          'phone_number': None,
                          'email': None,
                          'master_degree': None,
                          'bachelor_degree': None,
                          'phd_degree': None}
        return return_dict
    elif len(raw_txt) < 8:
        return_dict = {   'index' : index,
                          'name': None,
                          'phone_number': None,
                          'email': None,
                          'master_degree': None,
                          'bachelor_degree': None,
                          'phd_degree': None}
    else:
        info_dict = get_info(raw_txt)
        return_dict = {**{'index': index}, **info_dict}
        return return_dict

# execution ---------------------
folder = glob.glob(r"./nlp-coe/resume-parser/data/resume_samples/*")

for file_path in folder:
    print(parse_resume(file_path))

# -------------------- working process ---------------------------------------
def extract_graduation_date(txt1):
    dates=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    #/^(?=^abc)(?=.*xyz$)(?=.*123)(?=^(?:(?!456).)*$).*$/
    # Working to extract Months :x="(?=("+'|'.join(dates)+r"))"
    x="(?is)education.*?(\d{4})"
    # Working to extract year after education: x="(?is)education.*?(\d{4})"
    if len(re.findall(x,txt1))==0:
        return None
    return max(re.findall(x,txt1))
    ## for dates
    #for dt in dates:
    #    if dt in txt:
    #        return dt


def extract_phd_degree(sentence_txt):
    #https: // abbreviations.yourdictionary.com / articles / degree - abbreviations.html
    degree_list = ["Ph.D.", "M.S.", "MS", "Masters", "Master's", "MA", "M.A.", "MBA", "B.S.E.", "M.S.E.", "Bachelors", "B.tech", "Bachelor's", "Bachelor"]
    degree = {}
    words = nltk.word_tokenize(sentence_txt)
    phd=[]
    for j in range(len(words)):
        if j+2<len(words):
            if "Ph.D." in words[j]:
                phd.append(words[j] +" " + words[j+1] +" " + words[j+2])
    return phd

tagged = nlp(sample_txt) #assumption
degree_list = ["Ph.D.", "M.S.", "MS", "Masters", "Master's", "MA", "M.A.", "MBA", "B.S.E.", "M.S.E.", "Bachelors",
               "B.tech", "Bachelor's", "Bachelor"]
for d in degree_list: print(d)

for word in tagged.ents:
    #print(word.text, word.label_)

        #print(word.text)
        for d in degree_list:
            if d in word.text and word.label_ == "ORG":
                print(word.text)
        #temp_person.append(word.text)
    #return temp_person #return all names

def extract_mast_degree(txt):
    degree_list = ["Ph.D.","M.S.","MS","Masters","Master's","Bachelors","B.tech","Bachelor's","Bachelor"]
    words = nltk.word_tokenize(txt)
    mast=[]
    for j in range(len(words)):
        if j+2<len(words):
            if "M.S." in words[j] or "M.S" in words[j] or "MS" in words[j] or "Masters" in words[j] or "Master's" in words[j]:
                mast.append(words[j] +" " + words[j+1] +" " + words[j+2])
        if j+5<len(words):
            if "Master" in words[j]:
                mast.append(words[j] +" " + words[j+1] +" " + words[j+2] +" " + words[j+3] +" " + words[j+4] + " " + words[j+5])
    return mast

def extract_bach_degree(txt):
    degree_list = ["Ph.D.","M.S.","MS","Masters","Master's","Bachelors","B.tech","Bachelor's","Bachelor"]
    words = nltk.word_tokenize(txt)
    bach=[]
    for j in range(len(words)):
        if j+2<len(words):
            if "Bachelors" in words[j] or "B.tech" in words[j] or "Bachelor's" in words[j] or "B.A." in words[j] or "BS" in words[j] or "B.Sc." in words[j] or "B.S" in words[j]:
                bach.append(words[j] +" " + words[j+1] +" " + words[j+2])
        if j+5<len(words):
            if "Bachelor" in words[j]:
                bach.append(words[j] +" " + words[j+1] +" " + words[j+2] +" " + words[j+3] +" " + words[j+4] + " " + words[j+5])
    return bach

def get_degree_info(sentence_txt, tokens_txt): #-> list of person names
    temp_person = []
    nltk_tagged = pos_tag(tokens_txt[:11])
    for k in range(10):
        if nltk_tagged[k][1] == 'NNP' and nltk_tagged[k + 1][1] == 'NNP':
            nltk_name = nltk_tagged[k][0] + ' ' + nltk_tagged[k + 1][0]
            temp_person.append(nltk_name)
            if len(temp_person) > 0:# stop as long as you got the nltk_name
                return temp_person

        if k > 1:
            if (tokens_txt[k].lower() in indian_last_names) or (tokens_txt[k].lower() in chinese_last_names):
                j = k - 1
                return([tokens_txt[j] + " " + tokens_txt[k]]) #what if kristin (jiating) chen?

    tagged = nlp(sentence_txt[0:200]) #assumption
    for word in tagged.ents:
        if word.label_ == "PERSON":
            temp_person.append(word.text)
        return temp_person #return all names

def extract_master_school(txt):
    masters_set = set()
    p = re.compile(r'(School|University).*(Master|MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech).*,')
                   #'|((University).*(Master|MS|M\.S\.|MA|M\.A\.|MBA|M\.S\.E|M\.tech).*\\n)')  # greedy
    for m in re.finditer(p, txt):
        print(m.group())
        # candidate_txt = m.group().split(',')[0] #MS Excel MS word....
        # print(candidate_txt)
    #     if 'University' in candidate_txt:
    #         return ' '.join(candidate_txt.split()) #immediate return master
    #     else:
    #         masters_set.add(' '.join(candidate_txt.split()))
    # return masters_set
extract_master_school(sample_txt)