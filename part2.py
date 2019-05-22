import os, json, math

CORPUS_PATH = "corpus/derlem.txt"
punctuations = "\"!^%<+~*;:(?&}]|,')-#`@/$_{.>[\="
DOCUMENT_NUMBER = 0

df = {}
tf = {}
tf_idf = {}

def CalculateTfIdf(n):
    for paragraph_id in tf.keys():
        w = {} # score dictionary
        length = 0.0 # length of the tf.idf vector
        for word in df.keys():
            tf_ = 0
            idf = math.log10(n/df[word])
            
            if word in tf[paragraph_id]:
                tf_ = 1 + math.log10(tf[paragraph_id][word])
            
            score = tf_ * idf
            
            if score > 0.1: # threshold
                w[word] = score
                length += w[word]**2
        
        length = math.sqrt(length)
        
        for word in w.keys(): # normalization
            w[word] = w[word]/length 
        
        tf_idf[paragraph_id] = w # add it to the tf_idf

def CreateFiles():
    for f, fname in zip([tf,df,tf_idf],["tf.json", "df.json", "tf_idf.json"]):
        o = open(os.path.join("Output", fname), "w", encoding="utf-16")
        o.write(json.dumps(f, indent=4, sort_keys=True))
        o.close()

def Freq(paragraph):
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
    return term_freq

def ReadCorpus(path):
    global DOCUMENT_NUMBER
    corpus = open(path, "r", encoding="utf-16")
    for line in corpus:
        if line == "\n":
            continue
        for c in punctuations:
            line = line.replace(c, " ")
        line = line.lower()
        paragraph_id, paragraph = line.split(maxsplit=1)
        tf[paragraph_id] = Freq(paragraph)
        DOCUMENT_NUMBER += 1
    CalculateTfIdf(DOCUMENT_NUMBER)

def Sentence2TfIdf(s):
    tfidf_s = {}
    tf = {}
    idf = {}
    for c in punctuations:
        s = s.replace(c, " ")
    s = s.lower()
    word_count = len(s.split())
    
    for word in s.split():
        tf.setdefault(word, 0)
        tf[word] = tf[word] + 1
        if word in df.keys():
            idf[word] = math.log(DOCUMENT_NUMBER / df[word])
        else:
            idf[word] = math.log(DOCUMENT_NUMBER / 1)
    for word in tf.keys():
        tfidf_s[word] = tf[word] * idf[word]
    return tfidf_s
    




def cosine_prox(dict1, dict2): # inner product of two dictionaries
    sum = 0.0 
    for word in dict1.keys():
        if word in dict2:
            sum += dict1[word] * dict2[word]
    return sum

ReadCorpus(CORPUS_PATH)

# CreateFiles()

Q = input("Type your question (To stop type END)")

while Q != "END":
    scores = {} # scores of all movies
    
    for id in tf_idf.keys(): 
        scores[id] = cosine_prox(Sentence2TfIdf(Q), tf_idf[id])
    
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()

    print([x[0] for x in sorted_x[:10]])