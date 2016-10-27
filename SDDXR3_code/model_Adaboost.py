#coding=utf-8
#python2
import pandas as pd
data=pd.read_csv("feature_sd_07_35.csv",encoding="utf-8")
data.set_index("user",inplace=True)
print(data.shape)
y=data["is_buy"]
X=data.drop("is_buy",axis=1)
print("data is on")
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print("train_test data is on")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
print("training....")
def train(ntree,rate):
    model=AdaBoostClassifier(n_estimators=ntree,learning_rate=rate)
    model.fit(X_train,y_train)
    predict_con=model.predict(X_test)
    cm=confusion_matrix(y_test,predict_con,labels=[0,1])
    r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*r1*p1/(r1+p1)
    print "ntrees=%s,learning_rate=%s,recall1=%s,precision1=%s,f11=%s"%(ntree,rate,r1,p1,f11)

for ntree in [50,100,150,200,250,300]:
    for rate in [0.1,1,2,10]:
        train(ntree,rate)
print("done")
