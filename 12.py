import os, json, math
import sys
import nltk
import numpy as np

CORPUS_PATH = "corpus/derlem.txt"
WEIGHTS_PATH = 'weights/'
GLOVE_PATH = 'glove/'
FASTTEXT_PATH = 'ft/'
QA_PATH = sys.argv[1] + 'soru_gruplari.txt'
TASK1_PATH = sys.argv[2]
TASK2_PATH = sys.argv[3]

punctuations = "\"!^%<+~*;:(?&}]|,')-#`@/$_{.>[\="

def createFiles(weights_path, tf, df, tf_idf, doc_num, corpus_dict):
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

    return tf, df, tf_idf, doc_num, corpus_dict

def tokenize(text):
    punctuations = "\"!^%<+~*;:(?&}]|,)-#@/$_{.>[\="
    for c in punctuations:
        text = text.replace(c, " ")
    text = text.lower()

    response = []
    for word in text.split():
        if "'" in word:
            response.append(word.split("'")[0])
            #response += word.split("'")[0] + " "
        elif '’' in word:
            response.append(word.split('’')[0])
            #response += word.split('’')[0] + " "
        elif '`' in word:
            response.append(word.split('`')[0])
            #response += word.split('`')[0] + " "
        else:
            response.append(word)
    return " ".join(response)

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
        paragraph_id, paragraph = line.split(maxsplit=1)
        corpus_dict[paragraph_id] = paragraph
        paragraph = tokenize(paragraph)
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
    s = tokenize(s)
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
    #return 1 - spatial.distance.cosine(list1, list2)
    for word in dict1.keys():
        if word in dict2:
            result += dict1[word] * dict2[word]
    return result #/ (lenght1 * lenght2)

def cosine_similarity_normalized_dict(dict1, dict2): # inner product of two dictionaries
    result = 0.0 
    lenght1 = np.linalg.norm(list(dict1.values()))
    lenght2 = np.linalg.norm(list(dict2.values()))
    #return 1 - spatial.distance.cosine(list1, list2)
    for word in dict1.keys():
        if word in dict2:
            result += dict1[word] * dict2[word]
    return result / (lenght1 * lenght2)

def get_intersection(sentence1, sentence2):
    return list(set(tokenize(sentence1).split()) & set(tokenize(sentence2).split()))

def getBigrams(s):
    subset = []
    s = tokenize(s)
    for i in range(len(s)-1):
        subset.append(s[i:i+2])
    return subset

def intersect(list1,list2):
    return [value for value in list1 if value in list2]

def intersect_with_jaccard(list1, list2):
    intersection = []
    for word1 in list1:
        for word2 in list2:
            if jaccard_similarity(word1, word2) >= (min(len(word1), len(word2)) * 0.06):
                intersection.append(word1)
    return intersection

def union(list1, list2):
    return list(set(list1) | set(list2))

def jaccard_similarity(s1, s2):
    return len(intersect(getBigrams(s1),getBigrams(s2))) / len(union(getBigrams(s1),getBigrams(s2)))

def find_paragraph_dict(df, tf_idf, doc_num, sentence, return_number):
    scores = {}
    for key in tf_idf.keys(): 
        scores[key] = cosine_similarity_dict(sentence2TfIdf(df, doc_num, sentence), tf_idf[key])
    
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    return sorted_x[0:return_number]

def find_answer(corpus_dict, df, tf_idf, doc_num, question):
    print(question)
    paragraph_ids = find_paragraph_dict(df, tf_idf, doc_num, question, 25)
    question = tokenize(question)
    scores = {}
    for paragraph_id in [x[0] for x in paragraph_ids]:
        paragraph = corpus_dict[paragraph_id]
        sent_text = nltk.sent_tokenize(paragraph) # this gives us a list of sentences
        for sentence in sent_text:
            scores[sentence] = (cosine_similarity_dict(sentence2TfIdf(df, doc_num, sentence), sentence2TfIdf(df, doc_num, question)), paragraph_id) # * math.sqrt(cosine_similarity_dict(sentence2TfIdf(df, doc_num, question), tf_idf[paragraph_id]))
    sorted_x = sorted(scores.items(), key=lambda kv: kv[1])
    sorted_x.reverse()
    response = sorted_x[0][0]
    '''print(response)
    
    intersection = intersect_with_jaccard(nltk.word_tokenize(response), nltk.word_tokenize(tokenize(question)))
    for intersect in intersection:
        response = response.replace(intersect, ' ')
    response = tokenize(response)'''
    return remove_question(tokenize(response), tokenize(question), sorted_x[0][1][1])

def find_paragraphs_dict(df, tf_idf, qa, doc_num):
    t = 0
    f = 0
    for key in qa.keys():
        if qa[key][2] in [x[0] for x in find_paragraph_dict(df, tf_idf, doc_num, qa[key][0], 10)]:
            t = t + 1
        else:
            f = f + 1
        print('true' + str(t))
        print(f)

def remove_question(answer, question, paragraph_id):
    print(paragraph_id)
    answer = answer.split()
    question = question.split()
    response = [word for word in answer if tf_idf[paragraph_id][word] > 1.5]
    return " ".join(response)


tf_idf, tf, df, doc_num, corpus_dict = readCorpus(CORPUS_PATH)
qa = read_qa(QA_PATH)
#createFiles(WEIGHTS_PATH, tf, df, tf_idf, doc_num, corpus_dict)
tf, df, tf_idf, doc_num, corpus_dict = loadFiles(WEIGHTS_PATH)

#find_paragraphs_dict(df, tf_idf, qa, doc_num)
print(find_answer(corpus_dict, df, tf_idf, doc_num, qa['S907'][0]))
print(find_answer(corpus_dict, df, tf_idf, doc_num, qa['S41'][0]))
print(find_answer(corpus_dict, df, tf_idf, doc_num, qa['S2017'][0]))

#print(len(union(getBigrams('araba'),getBigrams('nehri'))))
#print(intersect_with_jaccard(['iklimi'], ['iklim']))
#print(jaccard_similarity('nehir', 'nehri'))



