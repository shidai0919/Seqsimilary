#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from keras.models import Sequential
model = Sequential()
# In[17]:
import pickle
import numpy as np
import jieba
import string
import pprint,pickle
f=open.('matchq.txt')
fm=open('word.pkl','rb')
lines = f.readlines()
sum=[]
for line in lines:
    match=[]
    if len(line.split("\t"))==3
        a=len(list(jieba.cut(line.split("\t")[0],cut_all=True)))
        b=len(list(jieba.cut(line.split("\t")[1],cut_all=True)))
        for wi in list(jieba.cut(line.split("\t")[0],cut_all=True)):
            if(data.has_key(wi)):

                for vi in list(jieba.cut(line.split("\t")[1],cut_all=True)):
                     if(data.has_key(vi)):
                         x=np.array(data[wi])
                         y=np.array(data[vi])
                         lw=np.sqrt(x.dot(x.T))
                         lv=np.sqrt(y.dot(y.T))
                         match.append(x.dot(y.T)/lw*lv)
                     else:
                         match.append(0)
            else:
                for vi in list(jieba.cut(line.split("\t")[1],cut_all=True)):
                    match.append(0)
        arr=np.array(match).reshape(a,b)
        print arr
        sum.append(arr)

output = open('d+d-array.pkl', 'wb')
pickle.dump(a, output)
output.close()
