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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def train(ntree):
    model=RandomForestClassifier(n_estimators=ntree,n_jobs=3,class_weight={0:2,1:1})
    model.fit(X_train,y_train)
    predict_con=model.predict(X_test)
    cm=confusion_matrix(y_test,predict_con,labels=[0,1])
    r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*r1*p1/(r1+p1)
    print "ntrees=%s,recall1=%s,precision1=%s,f11=%s"%(ntree,r1,p1,f11)

# for ntree in [10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000]:
#     train(ntree)

def dump_model(ntree=200):
    model=RandomForestClassifier(n_estimators=ntree,n_jobs=3,class_weight={0:2,1:1})
    model.fit(X_train,y_train)
    predict_con=model.predict(X_test)
    cm=confusion_matrix(y_test,predict_con,labels=[0,1])
    r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*r1*p1/(r1+p1)
    print "ntrees=%s,recall1=%s,precision1=%s,f11=%s"%(ntree,r1,p1,f11)
    with open("rf%s.model5"%ntree,"wb") as f:
        from pickle import dump
        dump(model,f)
        print("model dumped.")
dump_model(ntree=100)
dump_model(ntree=150)
dump_model(ntree=200)
