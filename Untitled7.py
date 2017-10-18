#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# In[17]:
import numpy as np
import jieba
import string
import pprint,pickle
f=open('1234.txt')


# In[18]:

import word2vec
fm=open('corpusWord2Vec.txt')


# In[16]:

lines = f.readlines()#读取全部内容
m=0
listVal = []
d={}
linea = fm.readlines()
for lineb in linea:
    l=[]
    for i in range(1,301):
         l.append(string.atof(lineb.split(" ")[i]))
    d[lineb.split(" ")[0].decode('utf-8')]=l
    print isinstance(lineb.split(" ")[0].decode('utf-8'),unicode)
for i in range(300):
   listVal.append(0);
aaalist=[]
j=0
dprobind={}
for line in lines:  
    aalist=[]
   # if(line.split("\t")[0]!=""and len(line.split("\t")[0])!=1):
        #print max(len(list(jieba.cut(line.split("\t")[0],cut_all=True))),m)
       # m=max(len(list(jieba.cut(line.split("\t")[0],cut_all=True))),m)
    for i in range(29):
        if len(list(jieba.cut(line.split("\t")[0],cut_all=True)))>i:
            if(d.has_key(list(jieba.cut(line.split("\t")[0],cut_all=True))[i])):
                print list(jieba.cut(line.split("\t")[0],cut_all=True))[i]
                aalist.append(d[list(jieba.cut(line.split("\t")[0],cut_all=True))[i]])
            else:
                aalist.append(listVal)
        else:
            aalist.append(listVal)
    aaalist.append(aalist)
    dprobind[j]=line.split("\t")[0]
    j+=1
#    x=np.array(aalist)
#    print x
#    np.savetxt("filename.txt",x)
        
output1=open('array01.pkl','wb')
pickle.dump(aaalist,output1)
output1.close()
pkl_file=open('array01.pkl','rb')
data1=pickle.load(pkl_file)
pprint.pprint(data1[0])
output2=open('dprobind.pkl','wb')
pickle.dump(dprobind,output2)
output2.close()
output3=open('word.pkl','wb')
pickle.dump(d,output3)
output3.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



