#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import jieba
import string
import pprint,pickle
import word2vec
import math
model = word2vec.load('corpusWord2Vec.bin')
dict={}
f=open('zhihu.pkl','rb')
data=pickle.load(f)
output3=open('dprobind.pkl','rb')
dprobind=pickle.load(output3)
inp= raw_input("Enter your input: ")
for ci in list(jieba.cut(inp,cut_all=True)):
    indexes = model.cosine(ci)
    j=0
    for index in indexes[0]:
	dict[index]=indexes[1][j]
        j+=1
limp=sorted(dict.items(), key=lambda e:e[1], reverse=True)
limp=limp[:10]
def listtodic(c):
    m=[]
    a={}
    z=0
    for x in c:
	a[x[0]]=x[1]
        z+=x[1]*x[1]
    m.append(a)
    m.append(math.sqrt(z))
    return m

#两个list求距离
def EuclideanDistances(c, b):
    s=0
    for key in c[0]:
	if(b[0].has_key(key)):
            s+=c[0][key]*b[0][key]
    s=s/c[1]*b[1]    
    return s	
		
		
		
dics={}		
j=0		
m=-100000.0		
for listi in data:
    dis=EuclideanDistances(listtodic(limp),listtodic(listi))
    m=max(dis,m)
    dics[dis]=j
    j+=1
x=dics[m]
strs=dprobind[x]
print strs
