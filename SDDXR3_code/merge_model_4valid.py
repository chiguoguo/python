# coding: utf-8
valid_file="feature_sd_07_79.csv"
lr_model="LR_2.model7"
print("models_level_1:",lr_model)

import pandas as pd
data=pd.read_csv(valid_file,encoding="utf-8")
data.set_index("user",inplace=True)
y=data["is_buy"]
X=data.drop("is_buy",axis=1)
X_level1=pd.DataFrame({})
print("test_data is on")
from pickle import load
models=["rf100.model5","rf200.model5","rf150.model5","gbrt200.model5","gbrt1000.model5"]
print("models_level_0:",len(models),models)
for i,m in enumerate(models):
    with open(m,"rb") as f:
        c_model=load(f)
        print("%s is on"%m)
    X_level1["m_%s"%i]=c_model.predict(X)
print("level_0 is built")
# from sklearn.cross_validation import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X_level1,y,test_size=0.3)


# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()
# model.fit(X_train,y_train)
# predict_con=model.predict(X_test)

print("validing...")
with open(lr_model,"rb") as f:
    lr_model=load(f)
    print("level_1:lr_model is on")
predict_con=lr_model.predict(X_level1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,predict_con,labels=[0,1])
precision1=float(cm[1][1])/(cm[1][1]+cm[0][1])
recall1=float(cm[1][1])/(cm[1][1]+cm[1][0])
f11=2*precision1*recall1/(precision1+recall1)
print cm
print "precison1=%s,recall1=%s,f11=%s"%(precision1,recall1,f11)
print("valid is done!")


