
# coding: utf-8

# In[1]:


#python2
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.pylab import *
mpl.rcParams['font.sans-serif']=['FangSong']
pd.set_option('display.width', 320)
pd.set_option('display.max_colwidth', -1)
pd.set_option('float_format','{:20,.2f}'.format)
from os import listdir
for d in listdir("."):
    print d.decode("gbk")


# In[2]:

data=pd.read_csv("feature_sd.csv",encoding="utf-8")
data.set_index("user",inplace=True)
print(data.shape)
data.head()


# In[3]:

y=data["is_buy"]
X=data.drop("is_buy",axis=1)
X.head()


# In[4]:

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# # Ridge

# In[20]:

from sklearn.linear_model import Ridge
model=Ridge()
model.fit(X_train,y_train)


# In[39]:

predict_con=model.predict(X_test)


# In[58]:

from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
ts,rs,ps,f1s=[],[],[],[]
for t in np.arange(0,1.1,0.05):
    threshold=t
    predict=[1 if p>threshold else 0 for p in predict_con]
#     print accuracy_score(y_test,predict)
    cm=confusion_matrix(y_test,predict,labels=[0,1])
#     print cm
    recall1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    precision1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*recall1*precision1/(recall1+precision1)
#     print threshold,recall1,precision1,f11
    ts.append(threshold)
    rs.append(recall1)
    ps.append(precision1)
    f1s.append(f11)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
plt.plot(ts,rs,label="recall1")
plt.plot(ts,ps,label="precision1")
plt.plot(ts,f1s,label="f11")
plt.legend()
plt.xlabel("threshold")
plt.ylabel("value")
plt.text(0.4,0.9,"Ridge Regression(Class 1)")
plt.show()


# # Lasso

# In[76]:

from sklearn.linear_model import Lasso
model=Lasso(0.1)
model.fit(X_train,y_train)
predict_con=model.predict(X_test)


# In[86]:

from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
ts,rs,ps,f1s=[],[],[],[]
for t in np.arange(0.3,0.5,0.005):
    threshold=t
    predict=[1 if p>threshold else 0 for p in predict_con]
#     print accuracy_score(y_test,predict)
    cm=confusion_matrix(y_test,predict,labels=[0,1])
#     print cm
    recall1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    precision1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*recall1*precision1/(recall1+precision1)
#     print threshold,recall1,precision1,f11
    ts.append(threshold)
    rs.append(recall1)
    ps.append(precision1)
    f1s.append(f11)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
plt.plot(ts,rs,label="recall1")
plt.plot(ts,ps,label="precision1")
plt.plot(ts,f1s,label="f11")
plt.legend()
plt.xlabel("threshold")
plt.ylabel("value")
plt.text(0.3,0.9,"Lasso Regression(Class 1,alpha=0.1)")
plt.show()


# # ElasticNet

# In[90]:

from sklearn.linear_model import ElasticNet
model=ElasticNet(0.1)
model.fit(X_train,y_train)


# In[91]:

predict_con=model.predict(X_test)


# In[94]:

from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
ts,rs,ps,f1s=[],[],[],[]
for t in np.arange(0.3,0.7,0.01):
    threshold=t
    predict=[1 if p>threshold else 0 for p in predict_con]
#     print accuracy_score(y_test,predict)
    cm=confusion_matrix(y_test,predict,labels=[0,1])
#     print cm
    recall1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    precision1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*recall1*precision1/(recall1+precision1)
#     print threshold,recall1,precision1,f11
    ts.append(threshold)
    rs.append(recall1)
    ps.append(precision1)
    f1s.append(f11)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
plt.plot(ts,rs,label="recall1")
plt.plot(ts,ps,label="precision1")
plt.plot(ts,f1s,label="f11")
plt.legend()
plt.xlabel("threshold")
plt.ylabel("value")
plt.text(0.3,0.9,"Elastic Regression(Class 1,alpha=0.1)")
plt.show()


# # LR

# In[95]:

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)


# In[96]:

predict_con=model.predict(X_test)


# In[101]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # LDA

# In[5]:

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model=LinearDiscriminantAnalysis()
model.fit(X_train,y_train)


# In[6]:

predict_con=model.predict(X_test)


# In[10]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # Bayesian

# In[11]:

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)


# In[12]:

predict_con=model.predict(X_test)


# In[15]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # RF

# In[17]:

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_jobs=3)
model.fit(X_train,y_train)


# In[18]:

predict_con=model.predict(X_test)


# In[21]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # RF2

# In[23]:

from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier(n_jobs=3)
model.fit(X_train,y_train)


# In[24]:

predict_con=model.predict(X_test)


# In[26]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # Adaboost

# In[28]:

from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier()
model.fit(X_train,y_train)


# In[29]:

predict_con=model.predict(X_test)


# In[32]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # GBRT

# In[36]:

from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(X_train,y_train)


# In[37]:

predict_con=model.predict(X_test)


# In[39]:

cm=confusion_matrix(y_test,predict_con,labels=[0,1])
print cm
r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
f11=2*r1*p1/(r1+p1)
print "recall1=%s,precision1=%s,f11=%s"%(r1,p1,f11)


# # SVM

# In[ ]:

from sklearn import svm
model=svm.SVC(n_jobs=3)
model.fit(X_train,y_train)


# In[ ]:

#on 244


# # Neural network

# In[6]:

from sklearn.neural_network import MLPClassifier
model=MLPClassifier()
model.fit(X_train,y_train)


# In[ ]:



