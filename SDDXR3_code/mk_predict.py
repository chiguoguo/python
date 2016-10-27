# coding: utf-8
predict_file="feature_sd_09_all.csvaa"
lr_model="LR_2.model7"
outfile="predict_91"
models=["rf100.model5","rf200.model5","rf150.model5","gbrt200.model5","gbrt1000.model5"]
print("models_level_1:",lr_model)

df_out_dict={"user":[],"is_buy":[]}
import pandas as pd
data=pd.read_csv(predict_file,encoding="utf-8")
df_out_dict["user"]=list(data["user"])
data.set_index("user",inplace=True)
y=data["is_buy"]
X=data.drop("is_buy",axis=1)
X_level1=pd.DataFrame({})
print("predict_data is on")
from pickle import load
print("models_level_0:",len(models),models)
for i,m in enumerate(models):
    with open(m,"rb") as f:
        c_model=load(f)
        print("%s is on"%m)
    X_level1["m_%s"%i]=c_model.predict(X)
print("level_0 is built")


print("predicting...")
with open(lr_model,"rb") as f:
    lr_model=load(f)
    print("level_1:lr_model is on")
predict_con=lr_model.predict(X_level1)
df_out_dict["is_buy"]=list(predict_con)

print("prediction mapped.dumping...")
df=pd.DataFrame(df_out_dict)
df.to_csv(outfile,encoding="utf-8",index=False)
print("all done!")


