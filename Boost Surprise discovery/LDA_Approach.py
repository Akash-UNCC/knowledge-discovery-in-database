# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import  pandas as pd

import os
import re
import gensim
from gensim import corpora
from gensim.models import word2vec
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import dictionary

from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full

##with open('traffic.csv', newline='') as csvfile:
     ##reader = csv.DictReader(csvfile)
##for row in reader:
##       print(row)

print(os.getcwd())
##cosine_similarity = np.dot(model['diabetes'], model['insulin'])/(np.linalg.norm(model['diabetes'])* numpy.linalg.norm(model['insulin']))

path = '/Users/sanket/Downloads/KDD/diabetes/diabetes'

##for filename in os.listdir(path):
countfile = 0
corpusDict=dict()
documents = []
writeDocf2 = open(os.getcwd()+"\\"+"diabetes"+"\\"+"Output.txt", 'a')
for filename in os.listdir(os.getcwd()+"/diabetes"):
    countfile = countfile + 1
    file = open(os.getcwd()+"\\"+"diabetes"+"\\"+filename,"r",encoding='utf8')
    filecontent=''
    for word in file:
        filecontent=filecontent+word +' '
    documents.append(filecontent)
    ##print(documents)
print('no of files =',countfile)
stoplist = set(stopwords.words('english'))
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
basetext=[]

for list in texts:
    for item in list:
        basetext.append(item)
#print(len(basetext))
#print(basetext)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=10)
bow_2 = lda.id2word.doc2bow(basetext)
lda_2 = lda[bow_2]
#print(lda_2)
for i in lda.print_topics(num_topics=5, num_words=10):
   # print(i[0])
    word=" ".join(re.findall("[A-Za-z]+.[A-Za-z]*", i[1]))
    word1 = word.replace("\"", "").strip().split()
    corpusDict[i[0]]=word1
    #for word3 in word1:
        #print(word3)
#print(corpusDict)

indvisualFileDict = dict()
documents = []
for filename in os.listdir(os.getcwd()+"/dib1"):
    file = open(os.getcwd()+"\\"+"dib1"+"\\"+filename,"r")
    texts=[]
    documents = []
    filecontent=''
    for word in file:
        filecontent=filecontent+word +' '
        documents.append(filecontent)
    stoplist = set(stopwords.words('english'))
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    basetext = []
    for list in texts:
        for item in list:
            basetext.append(item)
    bow_1 = lda.id2word.doc2bow(basetext)
    lda_1 = lda[bow_1]
    print("******************",filename)
    print("hellinger",hellinger(lda_1, lda_2))
    print("kullback_leibler",kullback_leibler(lda_1, lda_2))
    print("jaccard",jaccard(lda_1, lda_2))
    file.close()
        #dictionary = corpora.Dictionary(texts)
        #corpus = [dictionary.doc2bow(text) for text in texts]
    #lda1 = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=5)
#print(lda1)
#print(texts)
"""
basetext=[]
for list in texts:
    for item in list:
        basetext.append(item)
#print(len(basetext))
#print(basetext)

bow_1 = lda.id2word.doc2bow(basetext)
lda_1 = lda[bow_1]

#for i in  lda.print_topics(num_topics=1, num_words=5):
    #print(i)
    # word=" ".join(re.findall("[A-Za-z]+.[A-Za-z]*", i[1]))
    # word1 = word.replace("\"", "").strip().split()
    # indvisualFileDict[filename]=word1
    #documents=[]

print(lda_1)
print(hellinger(lda_1,lda_2))
print(kullback_leibler(lda_1,lda_2))
print(jaccard(lda_1,lda_2))
##for documents1 in documents:
  ##  print(documents1)

#print(lda.print_topics(num_topics=5, num_words=5))

#print(indvisualFileDict.values())
"""
"""
"""
"""
counter=0;
documents2=[]
for documents1 in documents:
    if(counter==1):
        documents1=''
        documents2.append(documents1)
        break
    counter = counter + 1
    documents2.append(documents1)


#print(len(documents2))


stoplist = set(stopwords.words('english'))
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents2]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, update_every=1, chunksize=10000, passes=5)

for i in  lda.print_topics(num_topics=1, num_words=5):
    print(i)
    word=" ".join(re.findall("[A-Za-z]+.[A-Za-z]*", i[1]))
    word1 = word.replace("\"", "").strip().split()
    for word3 in word1:
        print(word3)
"""