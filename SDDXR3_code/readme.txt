使用 modelR 来提取特征（提取训练集+测试集，提取预测集）

model_GBRT  生成一层模型 
model_RF    生成二层模型
model_Adaboost  #不用，不好

merge_model_lr.py  生成二层模型

mk_predict  应用到预测集


splitValiddata.py  切分数据（如果太大，或者将预测集切分为训练集）
phone_dict.pickle  沿用品牌价位模型的品牌价位字典
merge_output.py    合并输出（输入需要切割，因为加载不到内存）
choose_model.py    一些简单模型的尝试