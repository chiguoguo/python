
# coding: utf-8

# In[1]:


#python2
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.pylab import *
mpl.rcParams['font.sans-serif']=['FangSong']
pd.set_option('display.width', 320)
pd.set_option('display.max_colwidth', -1)
pd.set_option('float_format','{:20,.2f}'.format)
from os import listdir
for d in listdir("."):
    print d


# In[2]:

data=pd.read_csv("feature_sd_07_35.csv",encoding="utf-8")
data.set_index("user",inplace=True)
y=data["is_buy"]
X=data.drop("is_buy",axis=1)


# In[3]:

X_level1=pd.DataFrame({})


# In[4]:

with open("rf100.model5","rb") as f:
    from pickle import load
    rf100=load(f)
X_level1["rf100"]=rf100.predict(X)


# In[5]:

with open("rf150.model5","rb") as f:
    from pickle import load
    rf150=load(f)
X_level1["rf150"]=rf150.predict(X)


# In[6]:

with open("rf200.model5","rb") as f:
    from pickle import load
    rf200=load(f)
X_level1["rf200"]=rf200.predict(X)


# In[7]:

with open("gbrt200.model5","rb") as f:
    from pickle import load
    gbrt200=load(f)
X_level1["gbrt200"]=gbrt200.predict(X)


# In[8]:

with open("gbrt1000.model5","rb") as f:
    from pickle import load
    gbrt1000=load(f)
X_level1["gbrt1000"]=gbrt1000.predict(X)


# In[9]:

X_level1


# In[10]:

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_level1,y,test_size=0.3)


# In[21]:

# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()
# from sklearn.linear_model import LogisticRegressionCV
# model=LogisticRegressionCV(class_weight={1:1,0:2},n_jobs=10)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight={1:1,0:2},n_jobs=10)

model.fit(X_train,y_train)


# In[22]:

predict_con=model.predict(X_test)


# In[23]:

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict_con,labels=[0,1])
precision1=float(cm[1][1])/(cm[1][1]+cm[0][1])
recall1=float(cm[1][1])/(cm[1][1]+cm[1][0])
f11=2*precision1*recall1/(precision1+recall1)
print cm
print "precison1=%s,recall1=%s,f11=%s"%(precision1,recall1,f11)


# In[24]:

with open("LR_4.model7","wb") as f:
    from pickle import dump
    dump(model,f)


# In[25]:

model.coef_


# In[ ]:



