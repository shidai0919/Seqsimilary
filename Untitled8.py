#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# In[17]:
import numpy as np
import jieba
import string
import pprint,pickle
#f=open('1234.txt')


# In[18]:

import word2vec
fm=open('corpusWord2Vec.txt')


# In[16]:

#lines = f.readlines()#读取全部内容
#m=0
#listVal = []
d={}
linea = fm.readlines()
j=0
for lineb in linea:
#    for i in range(1,301):
#         l.append(string.atof(lineb.split(" ")[i]))
    d[lineb.split(" ")[0].decode('utf-8')]=j
    j+=1
#    print isinstance(lineb.split(" ")[0].decode('utf-8'),unicode)

output3=open('wordind.pkl','wb')
pickle.dump(d,output3)
output3.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



