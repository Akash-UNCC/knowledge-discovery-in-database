import numpy as np
import  pandas as pd
import csv
import csv
import os
import re
import csv
import gensim
from gensim import corpora
from gensim.models import word2vec
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import dictionary
import gensim
import glob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

##with open('traffic.csv', newline='') as csvfile:
     ##reader = csv.DictReader(csvfile)
##for row in reader:
##       print(row)


##cosine_similarity = np.dot(model['diabetes'], model['insulin'])/(np.linalg.norm(model['diabetes'])* numpy.linalg.norm(model['insulin']))

path = '/Users/sanket/Downloads/KDD/diabetes/diabetes'

##for filename in os.listdir(path):

corpusDict=dict()
documents = []
corp =''
corplist=[]
filecontent=''
blanklist=[]
stop_words = set(stopwords.words('english'))
corplist.append(blanklist)
for filename in os.listdir(os.getcwd()+"/diabetes"):
    file = open(os.getcwd() + "\\" + "diabetes" + "\\" + filename, "r", encoding='utf8')
    for line in file:
        corp = corp + line + ' '
corplist.append(corp.split())

#print('corplist',corplist)
#print(corplist)
dictionary = gensim.corpora.Dictionary(corplist)
#print(dictionary[5])
#print("Number of words in dictionary:",len(dictionary))
#for i in range(len(dictionary)):
    #print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in corplist]
#print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)

sims = gensim.similarities.Similarity(os.getcwd(),tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)
#print(type(sims))
#corp1=''
corplist1=[]
for filename in os.listdir(os.getcwd()+"/dib1"):
    file1 = open(os.getcwd()+"\\"+"dib1"+"\\"+filename,"r",encoding='utf8')
    corp1 = ''
    for line1 in file1:
        corp1 = corp1 + line1 + ' '

    corplist1=corp1.split()
    #print(corplist1)
    query_doc_bow = dictionary.doc2bow(corplist1)
    #print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    #print(query_doc_tf_idf)
    #print(query_doc_tf_idf)
    print('simialrity of ', filename,'is ',sims[query_doc_tf_idf])
