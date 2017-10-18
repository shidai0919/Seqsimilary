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
fm=open('word.pkl','rb')
data=pickle.load(fm)

# In[18]:




# In[16]:

lines = f.readlines()#读取全部内容
m=0
listVal = []

for line in lines:
    a=np.zeros(300)
    for ci in list(jieba.cut(line.split("\t")[0],cut_all=True)):
        if(data.has_key(ci)):
            a=a+np.array(data[ci])
#    a=a/float(len(list(jieba.cut(line.split("\t")[0],cut_all=True))))
    listVal.append(a.tolist())
#    x=np.array(aalist)
#    print x
#    np.savetxt("filename.txt",x)
        
output1=open('probvecno.pkl','wb')
pickle.dump(listVal,output1)
output1.close()



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



