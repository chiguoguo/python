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
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01)
print("train_test data is on")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
print("training....")
def train(ntree):
    model=GradientBoostingClassifier(n_estimators=ntree)
    model.fit(X_train,y_train)
    predict_con=model.predict(X_test)
    cm=confusion_matrix(y_test,predict_con,labels=[0,1])
    r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*r1*p1/(r1+p1)
    print "ntrees=%s,recall1=%s,precision1=%s,f11=%s"%(ntree,r1,p1,f11)

# for ntree in [100,200,300,400,500,600,700,800,900,1000,1100]:
#         train(ntree)
# print("done")


def dump_model(ntree=200):
    model=GradientBoostingClassifier(n_estimators=ntree)
    model.fit(X_train,y_train)
    predict_con=model.predict(X_test)
    cm=confusion_matrix(y_test,predict_con,labels=[0,1])
    r1=float(cm[1][1])/(cm[1][0]+cm[1][1])
    p1=float(cm[1][1])/(cm[0][1]+cm[1][1])
    f11=2*r1*p1/(r1+p1)
    print "ntrees=%s,recall1=%s,precision1=%s,f11=%s"%(ntree,r1,p1,f11)
    with open("gbrt%s.model5"%ntree,"wb") as f:
        from pickle import dump
        dump(model,f)
        print("model dumped.")
dump_model(ntree=1000)
dump_model(ntree=200)
dump_model(ntree=500)
