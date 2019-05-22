import os, json, math
import numpy as np

CORPUS_PATH = "corpus/derlem.txt"
WEIGHTS_PATH = 'weights/'
GLOVE_PATH = 'glove/'
QA_PATH = "corpus/soru_gruplari.txt"
punctuations = "\"!^%<+~*;:(?&}]|,')-#`@/$_{.>[\="

def createFiles(tf, df, tf_idf):
    for f, fname in zip([tf,df,tf_idf],["tf.json", "df.json", "tf_idf.json"]):
        o = open(os.path.join("weights", fname), "w", encoding="utf-16")
        o.write(json.dumps(f, indent=4, sort_keys=True))
        o.close()

def loadFiles(weights_path):
    with open(weights_path + 'tf.json') as f1:  
        tf = json.load(f1)
    with open(weights_path + 'df.json') as f2:  
        df = json.load(f2)
    with open(weights_path + 'tf_idf.json') as f3:  
        tf_idf = json.load(f3)
    return tf_idf, tf, df

def freq(df, paragraph):
    term_freq = {}
    for word in paragraph.split():
        if word in term_freq:
            term_freq[word] += 1
        else:
            if word in df:
                df[word] += 1
            else:
                df[word] = 1
            term_freq[word] = 1
    return term_freq, df

def calculateTfIdf(tf, df, doc_num):
    tf_idf = {}
    for paragraph_id in tf.keys():
        w = {} # score dictionary
        length = 0.0 # length of the tf.idf vector
        for word in df.keys():
            tf_ = 0
            idf = math.log10(doc_num/df[word])
            
            if word in tf[paragraph_id]:
                tf_ = 1 + math.log10(tf[paragraph_id][word])
            
            score = tf_ * idf
            w[word] = score
            '''
            length += w[word]**2
            
            if score > 0.1: # threshold
                w[word] = score
                
        
        length = math.sqrt(length)
        
        for word in w.keys(): # normalization
            w[word] = w[word]/length 
        '''
        tf_idf[paragraph_id] = w # add it to the tf_idf
    return tf_idf, tf, df, doc_num

def readCorpus(path):
    tf = {}
    df = {}
    doc_num = 0
    corpus = open(path, "r", encoding="utf-16")
    for line in corpus:
        if line == "\n":
            continue
        for c in punctuations:
            line = line.replace(c, " ")
        line = line.lower()
        paragraph_id, paragraph = line.split(maxsplit=1)
        tf[paragraph_id], df = freq(df, paragraph)
        doc_num += 1
    return calculateTfIdf(tf, df, doc_num)

def read_qa(path):
    qa = {}
    with open(path, "r", encoding="utf-16") as f:
        corpus = f.read()

    for qas in corpus.split("\n\n"):
        qas = qas.split("\n")
        answer = qas[len(qas) - 2]
        paragraph = qas[len(qas) - 1]
        for line in qas[:len(qas) - 2]:
            qa[line[:line.find(":")]] = [line[line.find(":") + 2:], answer[answer.find(":") + 2:], paragraph[paragraph.find(":") + 2:]]
    return qa

def sentence2TfIdf(df, doc_num, s):
    tfidf_s = {}
    tf = {}
    idf = {}
    for c in punctuations:
        s = s.replace(c, " ")
    s = s.lower()
    word_count = len(s.split())
    #length = 0.0
    for word in s.split():
        tf.setdefault(word, 0)
        tf[word] = tf[word] + 1
        if word in df.keys():
            idf[word] = math.log10(doc_num / df[word])
        
        else:
            idf[word] = math.log10(doc_num / 1)
        
    for word in idf.keys():
        #tfidf_s[word] = (1 + math.log(tf[word])) * idf[word]
        tfidf_s[word] = tf[word] * idf[word]
        #length += tfidf_s[word]**2
    
    #length = math.sqrt(length)

    for word in tfidf_s.keys(): # normalization
        tfidf_s[word] = tfidf_s[word]#/length 
    return tfidf_s
    
def cosine_similarity(dict1, dict2): # inner product of two dictionaries
    result = 0.0 
    '''
    lenght1 = np.linalg.norm(list(dict1.values()))
    lenght2 = np.linalg.norm(list(dict2.values()))
    '''
    for word in dict1.keys():
        if word in dict2:
            result += dict1[word] * dict2[word]
    return result #/ (lenght1 * lenght2)

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def find_paragraph(df, tf_idf, doc_num, sentence):
    scores = {}
    for key in tf_idf.keys(): 
        scores[key] = cosine_similarity(sentence2TfIdf(df, doc_num, sentence), tf_idf[key])
    
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    return sorted_x[0][0]
#vector = loadGloveModel(GLOVE_PATH + 'vectors.txt')
#print(vector['ve'])

tf_idf, tf, df, doc_num = readCorpus(CORPUS_PATH)
#tf_idf, tf, df = loadFiles(WEIGHTS_PATH)

qa = read_qa(QA_PATH)

t = 0
f = 0
print("qa time")

for key1 in qa.keys():
    scores = {}
    for key2 in tf_idf.keys(): 
        scores[key2] = cosine_similarity(sentence2TfIdf(df, doc_num, qa[key1][0]), tf_idf[key2])
    print(key1)
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    if sorted_x[0][0] == qa[key1][2]:
        t = t + 1
    else:
        f = f + 1

    print(t)
    print(f)