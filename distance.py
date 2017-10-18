#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# In[17]:
import numpy as np
import jieba
import string
import pprint,pickle
output1=open('array01.pkl','rb')
arrayf=pickle.load(output1)
output2=open('word.pkl','rb')
wordvec2=pickle.load(output2)
output3=open('dprobind.pkl','rb')
dprobind=pickle.load(output3)
inp= raw_input("Enter your input: ")
listVal=[]
for i in range(300):
   listVal.append(0);
aalist=[]
for i in range(29):
    if len(list(jieba.cut(inp,cut_all=True)))>i:
        if(wordvec2.has_key(list(jieba.cut(inp,cut_all=True))[i].decode('utf-8'))):
            print list(jieba.cut(inp,cut_all=True))[i]
            aalist.append(wordvec2[list(jieba.cut(inp,cut_all=True))[i].decode('utf-8')])
        else:
            aalist.append(listVal)
    else:
        aalist.append(listVal)
x=np.array(aalist)
def EuclideanDistances(A, B):
    c=A-B
    d=np.dot(c,np.transpose(c))
    e=np.trace(d)
    s=e**0.5
    return s
dics={}
j=0
m=100000.0
for listi in arrayf:
    volumn=np.array(listi)
    dis=EuclideanDistances(x,volumn)
    m=min(dis,m)
    dics[dis]=j
    j+=1
x=dics[m]
strs=dprobind[x]
print strs
