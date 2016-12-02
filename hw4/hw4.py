
# coding: utf-8

# In[ ]:

import string
from nltk.stem.porter import PorterStemmer
import word2phrase
from textblob import TextBlob
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import sys

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    s3 = s2.replace('_',' ')
    return s3

def get_book(fname):
    book_txt = open(fname).read().decode('utf-8')
    book_tb = TextBlob(book_txt.lower())
    return [s.words for s in book_tb.sentences if s]

def main(fname):
#    global two_word,three_word
    two_word=[]
    three_word=[]
    book_sentences = get_book(fname)
    phrased1 = word2phrase.train_model(book_sentences, min_count=2)
    phrased2 = word2phrase.train_model(phrased1, min_count=2)
    two_word_counter = Counter()
    three_word_counter = Counter()
    for sentence in phrased2:
        for word in sentence:
            if word.count('_') == 1:
                two_word_counter[word] += 1
                two_word.append(word)
            if word.count('_') == 2:
                three_word_counter[word] += 1
                three_word.append(word)
    
    return set(two_word),set(three_word)
    
def writePhrase(fname1,fname2):
#    global two_word,three_word
    two_word,three_word=main(fname1)
    with open(fname1, "r") as title1:
        with open(fname2, 'w') as title2:
            for line in title1:
                line=' '+line+' '
                tmp=''
                for x in three_word:
                    x=str(x)
                    x_=x.replace('_',' ')
                    while line.find(' '+x_+' ')>-1:
#                        line = line.replace(x_,x+" "+x_)
                        line = line.replace(x_,x)
                        if tmp.find(' '+x_+' ')==-1:
                            tmp=tmp+' '+x_+' '
                for x in two_word:
                    x=str(x)
                    x_=x.replace('_',' ')
                    while line.find(' '+x_+' ')>-1:
#                        line = line.replace(x_,x+" "+x_)
                        line = line.replace(x_,x)
                        if tmp.find(' '+x_+' ')==-1:
                            tmp=tmp+' '+x_+' '
#                line=line+tmp
                line=' '.join(line.split())+' \n'
                title2.write(line)
        
def rand_perm_mat(N):
    I = np.eye(N)
    p = (np.arange(N))
    return I[p]

#pre_process----------------------------------------------------------------------------------------------------
path_=sys.argv[1]
title_StackOverflow=path_+"/title_StackOverflow.txt"
check_index=path_+'/check_index.csv'
bow=sys.argv[2]

stemmer = PorterStemmer()
s_punc=string.punctuation+'´”’‘'.decode('utf-8')
stopwords = set([line.strip() for idx, line in enumerate(open('stopwords1.txt', 'r'))])
stopwords_ = [line.strip() for idx, line in enumerate(open('stopwords2.txt', 'r'))]
stopwords.update(stopwords_)    


with open(title_StackOverflow, "r") as title1:
    with open("title_processed_bow.txt", 'w') as title2:
        for line in title1:
            line=convert(line)
            
            line=' '+line+' '
            line=line.decode('utf-8').lower()
            line = line.replace("(ie "," i.e.")
            line = line.replace(" aja~x "," ajax ")

            line = line.replace("64 bits","64 bit")
            line = line.replace("32 bits","32 bit")
            line = line.replace("64bits","64 bit")
            line = line.replace("32bits","32 bit")
            line = line.replace("64bit","64 bit")
            line = line.replace("32bit","32 bit")
            
            line = line.replace(" vs 2003","visual studio 2003")
            line = line.replace(" vs 2005","visual studio 2005")
            line = line.replace(" vs 2008","visual studio 2008")
            line = line.replace(" vs 2010","visual studio 2010")
            
            line = line.replace("vs2003","visual studio 2003")
            line = line.replace("vs2005","visual studio 2005")
            line = line.replace("vs2008","visual studio 2008")
            line = line.replace("vs2010","visual studio 2010")
            
            line = line.replace("studio2005","studio 2005")
            line = line.replace("studio2008","studio 2008")
            
            
            #remove : !"'()*,-./:;<=>?@[\]_`{|}~&
            for c in s_punc:
                line = line.replace(c," ")
            
            line = line.replace(" i e "," i.e. ")
            line = line.replace(" v s "," vs ")
            
                       
            words_arr=line.split()
            words = [w for w in words_arr if w not in stopwords and w != '']
            line=' '.join(words)+' \n'  
            
            line_arr=line.split()
            singles = [stemmer.stem(x) for x in line_arr]
            line=' '.join(singles)+'\n'
            
            words_arr=line.split()
            words = [w for w in words_arr if w not in stopwords and w != '']
            new_line=' '+' '.join(words)+' \n'  
            
            new_line = new_line.replace(" 0 ","  0  ")
            new_line = new_line.replace(" 1 ","  1  ")
            new_line = new_line.replace(" 2 ","  2  ")
            new_line = new_line.replace(" 3 ","  3  ")
            new_line = new_line.replace(" 4 ","  4  ")
            new_line = new_line.replace(" 5 ","  5  ")
            new_line = new_line.replace(" 6 ","  6  ")
            new_line = new_line.replace(" 7 ","  7  ")
            new_line = new_line.replace(" 8 ","  8  ")
            new_line = new_line.replace(" 9 ","  9  ")
            
            new_line = new_line.replace(" 0 "," @ ")
            new_line = new_line.replace(" 1 "," @ ")
            new_line = new_line.replace(" 2 "," @ ")
            new_line = new_line.replace(" 3 "," @ ")
            new_line = new_line.replace(" 4 "," @ ")
            new_line = new_line.replace(" 5 "," @ ")
            new_line = new_line.replace(" 6 "," @ ")
            new_line = new_line.replace(" 7 "," @ ")
            new_line = new_line.replace(" 8 "," @ ")
            new_line = new_line.replace(" 9 "," @ ")
            
            new_line = new_line.replace(" s "," ")
            
            new_line=' '.join(new_line.split())+' \n'
            
            if len(new_line.split())==0:
                new_line='#'
                new_line=' '.join(new_line.split())+' \n'
                        
            new_line=new_line.encode("utf-8")
            
            title2.write(new_line)

writePhrase("title_processed_bow.txt","title_processed_phrase_bow.txt")

#feature_extraction-------------------------------------------------------------------------------------------------------

num_cluster=17

with open("title_processed_phrase_bow.txt") as f:
    content = f.read().splitlines()
t_words = re.findall(r'\w+', open('title_processed_phrase_bow.txt').read())
perfect_words=[x for x,v in Counter(t_words).most_common(500) if v <1000]
cv = CountVectorizer(vocabulary=perfect_words)
data=cv.fit_transform(content).toarray()
f_dim=len(perfect_words)
center_ = rand_perm_mat(f_dim)
new_center=center_[:num_cluster]
new_center=np.append(new_center,np.array([[0.0]*f_dim]),axis=0)
cluster = KMeans(n_clusters=len(new_center),init=new_center, n_jobs=4, verbose=1)
cluster.fit(data)

#test-----------------------------------------------------------------------------------------------------------------

chk_idx_csv = pd.read_csv(check_index).values
labels = cluster.labels_
result = [1 if labels[r[1]]<(len(new_center)-1) and labels[r[1]] == labels[r[2]] else 0 for r in chk_idx_csv]
result = np.array(result)
ids = np.array([i for i in xrange(5000000)])
np.savetxt(bow, np.column_stack((ids, result)),
        header='ID,Ans', comments='', fmt='%s', delimiter=',')

