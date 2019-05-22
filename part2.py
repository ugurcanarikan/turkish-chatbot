import os, json, math

CORPUS_PATH = "corpus/derlem.txt"

NUM_OF_PARAGRAPHS = 0

df = {}
tf = {}
tf_idf = {}

def CalculateTfIdf():
    for paragraph_id in tf.keys():
        w = {} # score dictionary
        length = 0.0 # length of the tf.idf vector
        for word in df.keys():
            tf_ = 0
            idf = math.log10(NUM_OF_PARAGRAPHS/df[word])
            
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
    corpus = open(path, "r", encoding="utf-16")
    for line in corpus:
        if line == "\n":
            continue
        paragraph_id, paragraph = line.split(maxsplit=1)
        tf[paragraph_id] = Freq(paragraph)
        NUM_OF_PARAGRAPHS += 1

ReadCorpus(CORPUS_PATH)
CalculateTfIdf()


