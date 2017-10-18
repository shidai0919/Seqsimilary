#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# In[17]:
import numpy as np
import jieba
import string
import pprint,pickle
f=open.('match.txt')
fm=open('word.pkl','rb')
lines = f.readlines()
for line in lines:
    match=[]
    if(line.split("\t").len()==3){
        a=list(jieba.cut(line.split("\t")[0],cut_all=True)).len()
        b=list(jieba.cut(line.split("\t")[1],cut_all=True)).len()
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
        match.reshape(a,b) 
    }
