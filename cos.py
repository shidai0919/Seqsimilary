#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# In[17]:
import numpy as np
import jieba
import string
import pprint,pickle

output2=open('word.pkl','rb')
wordvec2=pickle.load(output2)
f=open('probvecno.pkl','rb')
data=pickle.load(f)
output3=open('dprobind.pkl','rb')
dprobind=pickle.load(output3)
inp= raw_input("Enter your input: ")
a=np.zeros(300)
for ci in list(jieba.cut(inp,cut_all=True)):
    if(wordvec2.has_key(ci)):
        a=a+np.array(wordvec2[ci])
a=a/float(len(list(jieba.cut(inp,cut_all=True))))


def EuclideanDistances(c, b):
    s=-100000.0
    cl=np.sqrt(c.dot(c))
    bl=np.sqrt(b.dot(b))
    if(c.dot(b)!=0.0):
        s=c.dot(b)/(cl*bl)
    return s
dics={}
j=0
m=-100000.0
for listi in data:
    volumn=np.array(listi)
    dis=EuclideanDistances(a,volumn)
    m=max(dis,m)
    dics[dis]=j
    j+=1
x=dics[m]
strs=dprobind[x]
print strs
