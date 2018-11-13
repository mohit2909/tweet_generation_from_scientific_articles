
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential, Model,load_model
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from gensim.test.utils import common_texts, get_tmpfile
from keras.losses import categorical_crossentropy
from nltk.tokenize import  TweetTokenizer
import re
#keras.losses.categorical_crossentropy(y_true, y_pred)

tknz= TweetTokenizer()
model=load_model('my_model.h5')
dic={}
fl = open('indexer.txt','r')
for line in fl:
    line=re.sub('[\n]',"",line)
    line=line.split(': ')
    #print line
    dic[line[0]]=int(line[1])

fl.close()
sequences=list()
j=1
sum=0
fl=open('test_res.txt','r')
for line in fl:
    line=re.sub('[\n]',"",line)

    words=tknz.tokenize(line)
    sequences=list()
    seq=[]
    for word in words:
        if(word in dic):
            seq.append(dic[word])
    for i in range(len(seq)-2):
        sequences.append([seq[i],seq[i+2],seq[i+1]])
    sequences=array(sequences)
    if(len(sequences)<=1):
        continue
    y_true= sequences[:, -1]

    p=model.predict(sequences[:,:-1],verbose=0)
    ini=1
    for i in range(len(p)):
        prob= p[i][int(y_true[i])]
        if(prob!=0):
            ini= ini * prob
    num=1.0/len(y_true)
    sum+= pow(ini,num)
    print j
    j+=1;
    seq=[]

fl.close()
print (sum*1.0/j*1.0)
exit()
sequences=array(sequences)

X=sequences[:1000, :-1]
y_true= sequences[:1000, -1]
p=model.predict(X,verbose=0)
for i in range(len(p)):
    prob= p[i][int(y_true[i])]

    mx=max(row for row in p[i])
    print prob, mx

'''
y_pred=model.predict(X)
print (y_pred)
print (categorical_crossentropy(y_true,y_pred))
'''
