#a-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import jieba
import string
import pprint,pickle
import word2vec
model = word2vec.load('corpusWord2Vec.bin')
f=open('1234.txt')
lines = f.readlines()
allist=[]
v=0
for line in lines:
    dict={}
    for ci in list(jieba.cut(line.split("\t")[0],cut_all=True)):
        try:
            indexes = model.cosine(ci)
	    j=0
	    for index in indexes[0]:
	        dict[index]=indexes[1][j]
                j+=1
                v+=1
        except(KeyError),e:
            print v
            v+=1
            continue
    limp=sorted(dict.items(), key=lambda e:e[1], reverse=True)
    allist.append(limp[:10])
output1=open('zhihu.pkl','wb')
pickle.dump(allist,output1)
output1.close()
