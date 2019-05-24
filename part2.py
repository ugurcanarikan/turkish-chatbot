import os, json, math
import numpy as np
from scipy import spatial
import time
import gensim 

CORPUS_PATH = "corpus/derlem.txt"
WEIGHTS_PATH = 'weights/'
GLOVE_PATH = 'glove/'
FASTTEXT_PATH = 'ft/'
QA_PATH = "corpus/soru_gruplari.txt"
punctuations = "\"!^%<+~*;:(?&}]|,')-#`@/$_{.>[\="

def createFiles(weights_path, tf, df, tf_idf, doc_num, corpus_dict, corpus_embedding):
    with open(weights_path + 'tf.json', 'w+', encoding="utf-16") as f1:  
        json.dump(tf, f1)
    with open(weights_path + 'df.json', 'w+', encoding="utf-16") as f2:  
        json.dump(df, f2)
    with open(weights_path + 'tf_idf.json', 'w+', encoding="utf-16") as f3:  
        json.dump(tf_idf, f3)
    with open(weights_path + 'doc_num.json', 'w+', encoding="utf-16") as f4:  
        json.dump(doc_num, f4)
    with open(weights_path + 'corpus_dict.json', 'w+', encoding="utf-16") as f5:  
        json.dump(corpus_dict, f5)
    with open(weights_path + 'corpus_embedding.json', 'w+', encoding="utf-16") as f6:  
        json.dump(corpus_embedding, f6)

def loadFiles(weights_path):
    with open(weights_path + 'tf.json', encoding="utf-16") as f1:  
        tf = json.load(f1)
    with open(weights_path + 'df.json', encoding="utf-16") as f2:  
        df = json.load(f2)
    with open(weights_path + 'tf_idf.json', encoding="utf-16") as f3:  
        tf_idf = json.load(f3)
    with open(weights_path + 'doc_num.json', encoding="utf-16") as f4:  
        doc_num = json.load(f4)
    with open(weights_path + 'corpus_dict.json', encoding="utf-16") as f5:  
        corpus_dict = json.load(f5)
    with open(weights_path + 'corpus_embedding.json', encoding="utf-16") as f5:  
        corpus_embedding = json.load(f5)

    return tf, df, tf_idf, doc_num, corpus_dict, corpus_embedding

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
        for word in tf[paragraph_id].keys():
            tf_ = 0
            idf = math.log10(doc_num/df[word])
            
            if word in tf[paragraph_id].keys():
                tf_ = 1 + math.log10(tf[paragraph_id][word])
            
            score = tf_ * idf
            w[word] = score

        tf_idf[paragraph_id] = w # add it to the tf_idf
    return tf_idf, tf, df, doc_num

def readCorpus(path):
    tf = {}
    df = {}
    corpus_dict = {}
    doc_num = 0
    corpus = open(path, "r", encoding="utf-16")
    for line in corpus:
        if line == "\n":
            continue
        for c in punctuations:
            line = line.replace(c, " ")
        line = line.lower()
        paragraph_id, paragraph = line.split(maxsplit=1)
        corpus_dict[paragraph_id] = paragraph
        tf[paragraph_id], df = freq(df, paragraph)
        doc_num += 1
    tf_idf, tf, df, doc_num = calculateTfIdf(tf, df, doc_num)
    return tf_idf, tf, df, doc_num, corpus_dict

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
    
def cosine_similarity_dict(dict1, dict2): # inner product of two dictionaries
    result = 0.0 
    #lenght1 = np.linalg.norm(list(dict1.values()))
    #lenght2 = np.linalg.norm(list(dict2.values()))
    for word in dict1.keys():
        if word in dict2:
            result += dict1[word] * dict2[word]
    return result #/ (lenght1 * lenght2)

def cosine_similarity_list(list1, list2):
    if list1 == [0 for i in range(300)] or list2 == [0 for i in range(300)]:
        return 0
    return 1 - spatial.distance.cosine(list1, list2)

def get_intersection(sentence1, sentence2):
    for c in punctuations:
        sentence1 = sentence1.replace(c, " ")
        sentence2 = sentence2.replace(c, " ")
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    return list(set(sentence1.split()) & set(sentence2.split()))


def find_paragraph_dict(df, tf_idf, doc_num, sentence, return_number):
    scores = {}
    for key in tf_idf.keys(): 
        scores[key] = cosine_similarity_dict(sentence2TfIdf(df, doc_num, sentence), tf_idf[key])
    
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    return sorted_x[0:return_number]

def find_paragraph_list(corpus_embedding, sentence, return_number):
    scores = {}
    for key in corpus_embedding.keys(): 
        scores[key] = cosine_similarity_list(corpus_embedding[key], get_embeddings(model, sentence))
    
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    return sorted_x[0:return_number]

def find_paragraphs_dict(df, tf_idf, qa, doc_num):
    t = 0
    f = 0
    for key in qa.keys():
        if qa[key][2] in [x[0] for x in find_paragraph_dict(df, tf_idf, doc_num, qa[key][0], 1)]:
            t = t + 1
        else:
            f = f + 1
        print('true' + str(t))
        print(f)

def find_paragraphs_list(corpus_embedding, qa):
    t = 0
    f = 0
    print('started finding paragraphs')
    for key in qa.keys():
        if qa[key][2] in [x[0] for x in find_paragraph_list(corpus_embedding, qa[key][0], 1)]:
            t = t + 1
        else:
            f = f + 1
        print('finding paragraph of Q ' + str(list(qa.keys()).index(key)) + ' / ' + str(len(qa.keys())), end="\r")

        print('true' + str(t))
        print(f)

def get_embeddings(model, text):
    for word in sentence.split():
        word_embedding = [x for x in model[word]]
        embedding = [x + y for x, y in zip(token_embedding, embedding)]
    embedding = [x / len(sentence.tokens) for x in embedding]
    if embedding == [0 for i in range(300)]:
        print(text)
    return embedding

def get_corpus_embeddings(model, corpus_dict, tf_idf):
    corpus_embedding = {}
    for key in list(corpus_dict.keys()):
        embedding = [0.0 for i in range(300)]
        for word in corpus_dict[key].split():
            word_embedding = [x * tf_idf[key][word] for x in get_embeddings(model, word)]
            embedding = [x + y for x, y in zip(word_embedding, embedding)]
        embedding = [x / len(corpus_dict[key]) for x in embedding]
        corpus_embedding[key] = embedding
        print(str(list(corpus_dict.keys()).index(key)) + ' / ' + str(len(corpus_dict.keys())) + ' of document embeddings have been created ', end="\r")
    print('document embeddings have been created successfully')
    return corpus_embedding
'''
model = gensim.models.KeyedVectors.load_word2vec_format(FASTTEXT_PATH + 'cc.tr.300.vec', binary=False)
print(model['ve'])
print(type(model['ve']).__name__)
print(len(model['ve']))
'''
#tf_idf, tf, df, doc_num, corpus_dict = readCorpus(CORPUS_PATH)
#qa = read_qa(QA_PATH)
#corpus_embedding = get_corpus_embeddings(model, corpus_dict, tf_idf)
#createFiles(WEIGHTS_PATH, tf, df, tf_idf, doc_num, corpus_dict, corpus_embedding)
#tf, df, tf_idf, doc_num, corpus_dict, corpus_embedding = loadFiles(WEIGHTS_PATH)
#find_paragraph_list(corpus_embedding, qa['S2083'][0], 1)
#find_paragraph_list(corpus_embedding, qa['S1007'][0], 1)
#find_paragraphs_list(corpus_embedding, qa)
print(get_intersection('araba ve ev','sdf araba dsfsdcerc Araba, Ev'))










