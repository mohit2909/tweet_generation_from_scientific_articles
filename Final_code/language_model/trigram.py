from nltk.collocations import BigramCollocationFinder
import re
import codecs
import numpy as np
import string
import nltk
from math import log
from collections import Counter
#print(log2(5))
fl=open('processed_twitter_train.txt','r')
count=0
trigrams=[]
bigrams=[]
j=0
for row in fl:
    row= re.sub("[\n]","",row)
    tokens= row.split(" ")

    tp=[(tokens[i],tokens[i+1],tokens[i+2]) for i in range(0,len(tokens)-2)]
    for tgm in tp:
        trigrams.append(tgm)

    tp= [(tokens[i],tokens[i+1]) for i in range(0,len(tokens)-1)]
    for tgm in tp:
        bigrams.append(tgm)
    j+=1
tri_count = Counter(trigrams)
bi_count= Counter(bigrams)
print (count)
fl.close()
fl=open('processed_twitter_test.txt','r')
for row in fl:
    row= re.sub("[\n]","",row)
    sum=0
    print row
    tokens= row.split(" ")
    pr=[]
    for i in range(0,len(tokens)-2):
        if(bi_count[(tokens[i],tokens[i+1])]):
            print(bi_count[(tokens[i],tokens[i+1])])
            pr.append(max(tri_count[(tokens[i],tokens[i+1],tokens[i+2])]*1.0/bi_count[(tokens[i],tokens[i+1])],pow(10,-20)))
        else:
            pr.append(pow(10,-20))

    for r in pr:
        sum+=(r*log(r)/log(2))

    print(pow(2,sum))
fl.close()
