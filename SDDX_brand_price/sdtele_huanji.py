# -*- coding:UTF-8 -*-
from __future__ import division
from pyspark import SparkContext ,SparkConf
from operator import add

##本代码是对徐帅的山东电信换机项目的单机版代码的spark的分布式实现
##所需的文件有pi.txt,m2
#运行命令：spark-submit  ./sdtele_huanji.py

dictFile="phone_dict.pickle"#手机型号重新编码的字典文件
k_max=4#map6函数会用到此属性。
conf=SparkConf()
conf.setAppName("app_sddx")
conf.set("spark.executor.cores","16")
conf.set("spark.cores.max","576")
conf.set("spark.executor.memory","60g")
# conf.setMaster("local[4]")
conf.setMaster("spark://HadoopTest01:7077")

sc=SparkContext(conf=conf)
#small_m2是用于测试的小数据集。
# tf=sc.textFile("hdfs:///dogless/small_m2")
tf=sc.textFile("hdfs:///dogless/m3")
tf2=tf.map(lambda x:x.split(",")).map(lambda x:(x[0],(x[1],x[2],x[3])))

#################以下待删除####################
# def getChange(x):
#     ret=[]
#     for xi in sorted(x,key=lambda x:x[2],reverse=True):
#         if len(ret)<1 or int(ret[-1][2])-int(xi[2]) > 10: ret.append(xi)
#     return ",".join(["|".join(ri) for ri in ret])
#################以上待删除####################

def getChange(x):
    ret=[]
    history_phone=set()
    for xi in sorted(x,key=lambda x:x[2],reverse=False):
        if len(ret)<1 or xi[1] not in history_phone:
            ret.append(xi)
            history_phone.add(xi[1])
    return ",".join(["|".join(ri) for ri in ret[::-1]])

tf3=tf2.groupByKey().mapValues(getChange)
tf3.cache()#groupByKey所衔接的父子RDD为宽依赖关系
tf3.saveAsTextFile("/dogless/m2_out_tf3.txt")
tf4 = tf3.map(lambda x:str(x).strip().split(",")).map(lambda s:",".join([s[0].lstrip(r"u('").rstrip(r"'")]+["|".join(item.lstrip(r"u'").rstrip(r"')").split("|")[-2:]) for item in s[1:]]))
# tf4.saveAsTextFile("/dogless/m2_out_tf4.txt")
#5）parse2change.py 所做的自定义的机型价位代号代替机型
#standalone模式，我们可以直接读取本项目下的资源文件。phone_dict.pickle存入本项目。
with open(dictFile,"rb") as f:
  from pickle import load
  phone_dict=load(f)
broadcast_phone_dict = sc.broadcast(phone_dict)

def retype_brand(line):
    phone_dict = broadcast_phone_dict.value
    line=line.strip().split(",")
    phones=[line[0]] #phonenum
    to_write=[]
    for item in line[1:]:
        phone=item.split("|")[0]
        if phone in phone_dict:
            phones.append(phone_dict[phone])
        else:
            phones.append("Zc")
    to_write.append(",".join(phones))
    return to_write

tf5 = tf4.map(retype_brand)
# tf5.saveAsTextFile("/dogless/m2_out_tf5.txt")
##切分数据
tfs = tf5.randomSplit({0.9, 0.1},12345)
tf6_train = tfs[0]
tf6_test = tfs[1]
tf6_train.cache()
tf6_test.cache()
# tf6_test.saveAsTextFile("/dogless/m2_out_tf6.txt")


## 每行为"13611051353,甲,A,B,C,D"的数据通过map转换成(A_甲,1)，（A<B_甲,1），（A<B<C_甲,1），（A<B<C<D_甲,1）,然后reduce操作
def map6(listofline):
    line=listofline[0]
    line=line.strip().split(",")[:k_max+3]
    right=line[1]
    listoftuple=[()]
    if len(line)!=2:
        for i in range(3,len(line)+1):
            key="<".join(line[2:i])
            listoftuple.append((key,right))
    return listoftuple

def mapvalues_tf6(x):
    re_dict = dict()
    for xi in x:
        if(re_dict.has_key(xi)):
            #re_dict的xi的value+1
            re_dict[xi]+=1
        else:
            #re_dict添加值为xi的key，其value为1.
            re_dict[xi]=1
    #要对re_dict进行一些处理，re_dict中key的数量少于三个的不返回，并且将每个value除以所有value的总和。
    sum_value = 0
    for key in re_dict.iterkeys():
        sum_value+=re_dict[key]
    for key in re_dict.iterkeys():
        re_dict[key] = re_dict[key]/sum_value
    return re_dict


##tf7是类似这样的格式
tf7 = tf6_train.flatMap(map6).filter(lambda x : x!=()).groupByKey().mapValues(mapvalues_tf6).filter(lambda x : list(x)[-1]!=None)
tf7.cache()
# tf7.saveAsTextFile("/dogless/m2_out_tf7.txt")
list_tf7 = tf7.collect()
dict_for_model = dict(list_tf7)#我们得出了模型数据，接下来使用这个模型数据和测试集进行预测和验证。
bc = sc.broadcast(dict_for_model)

###################################################################################################
##在测试集上进行预测并检验。
###################################################################################################

############以下定义权重##############
alpha1=0.7
alpha2=0
alpha3=0
alpha4=0
alpha5=0
alpha6=0
weight_list = list()
weight_list.append(1)
weight_list.append(alpha1)
weight_list.append(alpha2)
weight_list.append(alpha3)
weight_list.append(alpha4)
weight_list.append(alpha5)
weight_list.append(alpha6)
bc_weight_list = sc.broadcast(weight_list)
###############以上定义权重################
##利用dict_for_model进行对测试集预测，并测试。
def map6_test(listofline):
    dict_for_model = bc.value
    weight_list_from_bc = bc_weight_list.value
    line=listofline[0]
    line=line.strip().split(",")[:k_max+3]
    list=[]
    predictions = dict()
    if len(line)!=2:
        for i in range(3,len(line)+1):
            key="<".join(line[2:i])
            list.append(key)
        ##接下来利用list中的key们结合dict_for_model进行预测，哪个值大，我们就认为哪个是最终的预测值。
        biggest_predict = dict()
        for i in range(0,len(list)):
            key = list[i]
            if dict_for_model.has_key(key):
                dict4select = dict_for_model[key]
                for key,value in dict4select.items():
                    if biggest_predict.has_key(key):
                        biggest_predict[key] += float(value)*float(weight_list_from_bc[i])
                    else:
                        biggest_predict[key] = float(value)*float(weight_list_from_bc[i])
        ##获取biggest_predict（类型为dict）中最大的value所对应的key。
        bpkeys = biggest_predict.keys()
        bp_value = 0
        final_pred = "None"
        for bpkey in bpkeys:
            tmp_value = biggest_predict[bpkey]
            if tmp_value > bp_value:
                bp_value = tmp_value
                final_pred = bpkey
        # final_pred = (k for k, v in biggest_predict.items() if v == max(biggest_predict.values()))
        if isinstance(final_pred,str):
            if(line[1] == final_pred):
                str_re = line[1]+"=="+final_pred
                return (line[0],True)
            else:
                str_re = line[1]+"!="+str(final_pred)
                return (line[0],False)
        else:
            return (line[0],"final_pred is not string")
        # return (k, v for k,v in biggest_predict.items() if v==max(biggest_predict.values()))

#tf8是测试集数据产生的结果。
# tf8 = tf6_test.map(map6_test)
# # tf8.saveAsTextFile("/dogless/m2_out_tf8.txt")
# local_list_tf = tf8.collect()
# right_count = 0
# total_count = 0
# for ele in local_list_tf:
#     if isinstance(ele,tuple):
#         print("tuple : "+str(ele[0])+str(ele[1]))
#         if ele[1]==True:
#             right_count+=1
#             total_count+=1
#         else:
#             total_count+=1
#     elif type(ele)==None:
#         print(ele)
#         # total_count+=1
#     else:
#         print("else")
#         # total_count+=1
# print("################################################################")
# print("right_count = "+str(right_count))
# print("total_count = "+str(total_count))
# print("recall_rate = "+str(right_count/total_count))
###################################################################################################
##使用所有数据tf5进行预测。
###################################################################################################
def map_predict(listofline):
    dict_for_model = bc.value
    weight_list_from_bc = bc_weight_list.value
    line=listofline[0]
    line=line.strip().split(",")[:k_max+3]
    list=[]
    predictions = dict()
    for i in range(2,len(line)+1):
        key="<".join(line[1:i])
        list.append(key)
    ##接下来利用list中的key们结合dict_for_model进行预测，哪个值大，我们就认为哪个是最终的预测值。
    biggest_predict = dict()
    for i in range(0,len(list)):
        key = list[i]
        if dict_for_model.has_key(key):
            dict4select = dict_for_model[key]
            for key,value in dict4select.items():
                if biggest_predict.has_key(key):
                    biggest_predict[key] += float(value)*float(weight_list_from_bc[i])
                else:
                    biggest_predict[key] = float(value)*float(weight_list_from_bc[i])
    ##获取biggest_predict（类型为dict）中最大的value所对应的key。
    bpkeys = biggest_predict.keys()
    bp_value = 0
    final_pred = "None"
    for bpkey in bpkeys:
        tmp_value = biggest_predict[bpkey]
        if tmp_value > bp_value:
            bp_value = tmp_value
            final_pred = bpkey
    # final_pred = (k for k, v in biggest_predict.items() if v == max(biggest_predict.values()))
    if isinstance(final_pred,str):
        if(line[1] == final_pred):
            str_re = str(final_pred)
            return (line[0],str_re)
        else:
            str_re = str(final_pred)
            return (line[0],str_re)
    else:
        return (line[0],"None")
##进行实际预测
tf_predict = tf5.map(map_predict)
# tf_predict.saveAsTextFile("/dogless/m2_out_tf_predict.txt")