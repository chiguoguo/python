# -*- coding:UTF-8 -*-
from pyspark import SparkContext, SparkConf
from pyspark.sql import *
import pyspark
from pyspark.ml.classification import *
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy
import datetime
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')
##离网状态为1.0表示已经离网，0.0表示未离网。
##用于获取8月离网用户的1到7月的状态信息。
##运行命令 nohup spark-submit  ./sdtele_lwyj_train_test.py >train_test_log &

APP_NAME = "hefei_lwyj_train_test"
##存储在hdfs上的8月离网用户数据文件。
loss_201608 = "/shandongdianxin/edw_m_pr_pri_loss_user.txt"
##存储在hdfs上的4~7月的离网用户信息。
loss_201604_07 = "/shandongdianxin/edw_m_pr_pri_loss_user2.txt"
##存储1~7月用户信息的文件
user_history_info_201601 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_01.txt"
user_history_info_201602 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_02.txt"
user_history_info_201603 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_03.txt"
user_history_info_201604 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_04.txt"
user_history_info_201605 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_05.txt"
user_history_info_201606 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_06.txt"
user_history_info_201607 = "/shandongdianxin/liwangyujing_user_history_info/temp_liuxw_0705_07.txt"

##临时表的表名
table_loss_201608 = "table_loss_201608"
table_loss_201607 = "table_loss_201607"
table_user_info_201601 = "table_user_info_201601"
table_user_info_201602 = "table_user_info_201602"
table_user_info_201603 = "table_user_info_201603"
table_user_info_201604 = "table_user_info_201604"
table_user_info_201605 = "table_user_info_201605"
table_user_info_201606 = "table_user_info_201606"
table_user_info_201607 = "table_user_info_201607"
tmp_table_m56 = "tmp_table_m56"
tmp_table_m456 = "tmp_table_m456"
tmp_table_m567 = "tmp_table_m567"
tmp_table_m67 = "tmp_table_m67"
table_df_for_train = "table_df_for_train"
table_df_for_test = "table_df_for_test"


conf = SparkConf().setAppName(APP_NAME)
conf.setMaster("spark://HadoopTest01:7077")
sc=SparkContext(conf=conf)
#声明HiveContext
hiveContext = pyspark.HiveContext(sparkContext=sc)
##############以下读取7月离网用户##############
rdd_loss_201604_07 = sc.textFile(loss_201604_07)
##下面这一行可以留作以后优化，3个map函数可以统一成一个。
rdd_loss_201607 = rdd_loss_201604_07.map(lambda line:line.split("|")).map(lambda x:(x[0],x[1])).filter(lambda n:n[0]=="201607")#.map(lambda m:m[1])
rdd_loss_201607_2 = rdd_loss_201607.map(lambda p : Row(uid=p[1],leave=1.0))##1.0表示离网
df_loss_201607 = hiveContext.createDataFrame(rdd_loss_201607_2)#df_loss_201607这个dataframe是7月的离网用户.
df_loss_201607.registerTempTable(table_loss_201607)
# df_loss_201607.show(33,False)
# something_m7 = rdd_loss_201607.collect()
# set_loss_m7 = set(something_m7)
# bc_m7_loss = sc.broadcast(set_loss_m7)
##############以上读取7月离网用户##############

# ##用于在4,5,6,7月的用户信息数据中筛选属于七月份离网用户的function。
# def filter_by_set_m7(x):
#     set_loss = bc_m7_loss.value
#     if x[0] in set_loss:
#         return True
#     else:
#         return False

##############以下读取6月用户信息##############
rdd_user_info_month6 = sc.textFile(user_history_info_201606)
rdd_month6_2 = rdd_user_info_month6.map(lambda line:line.split("|"))
##revenue:出账收入, data:总流量, wlan:wlan流量, two_g:2g总流量 , three_g: 3g总流量,four_g:4g总流量  ,ncall: 主叫通话次数, tcall:主叫通话时长  , ncalled:被叫通话次数
##   , tcalled:被叫通话时长 , ncall_loc:本地主叫通话次数 ,ncalled_loc:本地被叫通话次数 ,tcall_loc:本地主叫通话时长 ,tcalled_loc:本地被叫通话时长 ,
## tcall_roam:漫游主叫通话时长  ,tcalled_roam:漫游被叫通话时长 ,ncall_roam:漫游主叫通话次数 ,ncalled_roam:漫游被叫通话次数
## nconv_roam_p:省内漫游通话次数 ,nconv_roam_c:国内漫游通话次数 ,nconv_roam_i:国际漫游通话次数 ,tconv_roam_p:省内漫游通话时长 ,tconv_roam_c: 国内漫游通话时长 ,tconv_roam_i:国际漫游通话时长 ,
## nmsgup:上行短信条数 ,nmsgdn:下行短信条数
rdd_month6_3 = rdd_month6_2.map(lambda p : Row(uid=p[0],date=p[4]
                                                , revenue=float('0' if (p[5] is None or p[5]=='') else p[5]), data=float('0' if (p[6] is None or p[6]=='') else p[6])
                                               , wlan=float('0' if (p[7] is None or p[7]=='') else p[7]), two_g=float('0' if (p[8] is None or p[8]=='') else p[8])
                                               , three_g=float('0' if (p[9] is None or p[9]=='') else p[9]), four_g=float('0' if (p[10] is None or p[10]=='') else p[10])
                                                , ncall=float('0' if (p[11] is None or p[11]=='') else p[11]), tcall=float('0' if (p[12] is None or p[12]=='') else p[12])
                                               , ncalled=float('0' if (p[13] is None or p[13]=='') else p[13]), tcalled=float('0' if (p[14] is None or p[14]=='') else p[14])
                                               , ncall_loc=float('0' if (p[15] is None or p[15]=='') else p[15]), ncalled_loc=float('0' if (p[16] is None or p[16]=='') else p[16])
                                               , tcall_loc=float('0' if (p[17] is None or p[17]=='') else p[17]), tcalled_loc=float('0' if (p[18] is None or p[18]=='') else p[18])
                                               ,tcall_roam=float('0' if (p[19] is None or p[19]=='') else p[19]), tcalled_roam=float('0' if (p[20] is None or p[20]=='') else p[20])
                                               , ncall_roam=float('0' if (p[21] is None or p[21]=='') else p[21]), ncalled_roam=float('0' if (p[22] is None or p[22]=='') else p[22])
                                               , nconv_roam_p=float('0' if (p[23] is None or p[23]=='') else p[23]), nconv_roam_c=float('0' if (p[24] is None or p[24]=='') else p[24])
                                               , nconv_roam_i=float('0' if (p[25] is None or p[25]=='') else p[25]) , tconv_roam_p=float('0' if (p[26] is None or p[26]=='') else p[26])
                                              , tconv_roam_c=float('0' if (p[27] is None or p[27]=='') else p[27]), tconv_roam_i=float('0' if (p[28] is None or p[28]=='') else p[28])
                                               , nmsgup=float('0' if (p[29] is None or p[29]=='') else p[29]), nmsgdn=float('0' if (p[30] is None or p[30]=='') else p[30])))
df_month06_1 = hiveContext.createDataFrame(rdd_month6_3)
# print("###################################  month6  ###################################################")
# df_month06_1.show(19,False)
# df_month06_1.printSchema()
df_month06_1.registerTempTable(table_user_info_201606)
##############以上读取6月用户信息##############



##############以下读取5月用户信息##############
rdd_user_info_month5 = sc.textFile(user_history_info_201605)
rdd_month5_2 = rdd_user_info_month5.map(lambda line:line.split("|"))
##revenue:出账收入, data:总流量, wlan:wlan流量, two_g:2g总流量 , three_g: 3g总流量,four_g:4g总流量  ,ncall: 主叫通话次数, tcall:主叫通话时长  , ncalled:被叫通话次数
##   , tcalled:被叫通话时长 , ncall_loc:本地主叫通话次数 ,ncalled_loc:本地被叫通话次数 ,tcall_loc:本地主叫通话时长 ,tcalled_loc:本地被叫通话时长 ,
## tcall_roam:漫游主叫通话时长  ,tcalled_roam:漫游被叫通话时长 ,ncall_roam:漫游主叫通话次数 ,ncalled_roam:漫游被叫通话次数
## nconv_roam_p:省内漫游通话次数 ,nconv_roam_c:国内漫游通话次数 ,nconv_roam_i:国际漫游通话次数 ,tconv_roam_p:省内漫游通话时长 ,tconv_roam_c: 国内漫游通话时长 ,tconv_roam_i:国际漫游通话时长 ,
## nmsgup:上行短信条数 ,nmsgdn:下行短信条数
rdd_month5_3 = rdd_month5_2.map(lambda p : Row(uid=p[0]
                                                , revenue=float('0' if (p[5] is None or p[5]=='') else p[5]), data=float('0' if (p[6] is None or p[6]=='') else p[6])
                                               , wlan=float('0' if (p[7] is None or p[7]=='') else p[7]), two_g=float('0' if (p[8] is None or p[8]=='') else p[8])
                                               , three_g=float('0' if (p[9] is None or p[9]=='') else p[9]), four_g=float('0' if (p[10] is None or p[10]=='') else p[10])
                                                , ncall=float('0' if (p[11] is None or p[11]=='') else p[11]), tcall=float('0' if (p[12] is None or p[12]=='') else p[12])
                                               , ncalled=float('0' if (p[13] is None or p[13]=='') else p[13]), tcalled=float('0' if (p[14] is None or p[14]=='') else p[14])
                                               , ncall_loc=float('0' if (p[15] is None or p[15]=='') else p[15]), ncalled_loc=float('0' if (p[16] is None or p[16]=='') else p[16])
                                               , tcall_loc=float('0' if (p[17] is None or p[17]=='') else p[17]), tcalled_loc=float('0' if (p[18] is None or p[18]=='') else p[18])
                                               ,tcall_roam=float('0' if (p[19] is None or p[19]=='') else p[19]), tcalled_roam=float('0' if (p[20] is None or p[20]=='') else p[20])
                                               , ncall_roam=float('0' if (p[21] is None or p[21]=='') else p[21]), ncalled_roam=float('0' if (p[22] is None or p[22]=='') else p[22])
                                               , nconv_roam_p=float('0' if (p[23] is None or p[23]=='') else p[23]), nconv_roam_c=float('0' if (p[24] is None or p[24]=='') else p[24])
                                               , nconv_roam_i=float('0' if (p[25] is None or p[25]=='') else p[25]) , tconv_roam_p=float('0' if (p[26] is None or p[26]=='') else p[26])
                                              , tconv_roam_c=float('0' if (p[27] is None or p[27]=='') else p[27]), tconv_roam_i=float('0' if (p[28] is None or p[28]=='') else p[28])
                                               , nmsgup=float('0' if (p[29] is None or p[29]=='') else p[29]), nmsgdn=float('0' if (p[30] is None or p[30]=='') else p[30])))
df_month05_1 = hiveContext.createDataFrame(rdd_month5_3)
# print("###################################  month5  ###################################################")
# df_month05_1.show(20)
# df_month05_1.printSchema()
# df_month05_1 = df_month05_1.repartition(1)
df_month05_1.registerTempTable(table_user_info_201605)
##############以上读取5月用户信息##############

##############以下读取4月用户信息##############
rdd_user_info_month4 = sc.textFile(user_history_info_201604)
rdd_month4_2 = rdd_user_info_month4.map(lambda line:line.split("|"))
##revenue:出账收入, data:总流量, wlan:wlan流量, two_g:2g总流量 , three_g: 3g总流量,four_g:4g总流量  ,ncall: 主叫通话次数, tcall:主叫通话时长  , ncalled:被叫通话次数
##   , tcalled:被叫通话时长 , ncall_loc:本地主叫通话次数 ,ncalled_loc:本地被叫通话次数 ,tcall_loc:本地主叫通话时长 ,tcalled_loc:本地被叫通话时长 ,
## tcall_roam:漫游主叫通话时长  ,tcalled_roam:漫游被叫通话时长 ,ncall_roam:漫游主叫通话次数 ,ncalled_roam:漫游被叫通话次数
## nconv_roam_p:省内漫游通话次数 ,nconv_roam_c:国内漫游通话次数 ,nconv_roam_i:国际漫游通话次数 ,tconv_roam_p:省内漫游通话时长 ,tconv_roam_c: 国内漫游通话时长 ,tconv_roam_i:国际漫游通话时长 ,
## nmsgup:上行短信条数 ,nmsgdn:下行短信条数
rdd_month4_3 = rdd_month4_2.map(lambda p : Row(uid=p[0]
                                                , revenue=float('0' if (p[5] is None or p[5]=='') else p[5]), data=float('0' if (p[6] is None or p[6]=='') else p[6])
                                               , wlan=float('0' if (p[7] is None or p[7]=='') else p[7]), two_g=float('0' if (p[8] is None or p[8]=='') else p[8])
                                               , three_g=float('0' if (p[9] is None or p[9]=='') else p[9]), four_g=float('0' if (p[10] is None or p[10]=='') else p[10])
                                                , ncall=float('0' if (p[11] is None or p[11]=='') else p[11]), tcall=float('0' if (p[12] is None or p[12]=='') else p[12])
                                               , ncalled=float('0' if (p[13] is None or p[13]=='') else p[13]), tcalled=float('0' if (p[14] is None or p[14]=='') else p[14])
                                               , ncall_loc=float('0' if (p[15] is None or p[15]=='') else p[15]), ncalled_loc=float('0' if (p[16] is None or p[16]=='') else p[16])
                                               , tcall_loc=float('0' if (p[17] is None or p[17]=='') else p[17]), tcalled_loc=float('0' if (p[18] is None or p[18]=='') else p[18])
                                               ,tcall_roam=float('0' if (p[19] is None or p[19]=='') else p[19]), tcalled_roam=float('0' if (p[20] is None or p[20]=='') else p[20])
                                               , ncall_roam=float('0' if (p[21] is None or p[21]=='') else p[21]), ncalled_roam=float('0' if (p[22] is None or p[22]=='') else p[22])
                                               , nconv_roam_p=float('0' if (p[23] is None or p[23]=='') else p[23]), nconv_roam_c=float('0' if (p[24] is None or p[24]=='') else p[24])
                                               , nconv_roam_i=float('0' if (p[25] is None or p[25]=='') else p[25]) , tconv_roam_p=float('0' if (p[26] is None or p[26]=='') else p[26])
                                              , tconv_roam_c=float('0' if (p[27] is None or p[27]=='') else p[27]), tconv_roam_i=float('0' if (p[28] is None or p[28]=='') else p[28])
                                               , nmsgup=float('0' if (p[29] is None or p[29]=='') else p[29]), nmsgdn=float('0' if (p[30] is None or p[30]=='') else p[30])))
df_month04_1 = hiveContext.createDataFrame(rdd_month4_3)
# print("###################################  month4  ###################################################")
# df_month04_1.show(21,False)
# df_month04_1.printSchema()
df_month04_1.registerTempTable(table_user_info_201604)
##############以上读取4月用户信息##############

##############以下4,5,6月用户信息进行特征处理##############
##5,6月的表进行关联查询。（注意，不可以进行三表关联查询，笛卡尔积太大）
list_col = ["revenue","data","wlan","two_g","three_g","four_g","ncall","tcall","ncalled","tcalled","ncall_loc","ncalled_loc","tcall_loc","tcalled_loc",
            "tcall_roam","tcalled_roam","ncall_roam","ncalled_roam","nconv_roam_p","nconv_roam_c","nconv_roam_i","tconv_roam_p","tconv_roam_c","tconv_roam_i",
            "nmsgup","nmsgdn"]
# m56_sql_part1 = "select m6.uid, m6.date, "  #暂时先不加日期。
m56_sql_part1 = "select m6.uid, "
m56_sql_part2 = ""
##sql中表示某字段为null的时候不要用=null ，而是用 is null.
for i in range(0,25):
    m56_sql_part2 += "m6."+list_col[i] +" as m6_"+list_col[i] +", m5."+list_col[i] +" as m5_"+list_col[i] +", (m6."+list_col[i] +"+m5."+list_col[i] +")/2 as  m6_m5_"+list_col[i] +"_avg, "
    m56_sql_part2 += "(m6."+list_col[i] +"+m5."+list_col[i] +") as m6_m5_"+list_col[i] +"_sum, (m6."+list_col[i] +" - m5."+list_col[i] +") as m6_m5_"+list_col[i] +"_diff, "
    m56_sql_part2 += "(case when m5."+list_col[i] +" = 0.0 then 0.0 else  (m6."+list_col[i] +"/m5."+list_col[i] +") end) as m6_m5_"+list_col[i] +"_rate,"
m56_sql_part2 = m56_sql_part2[0:-1]
m56_sql = m56_sql_part1+m56_sql_part2
df_month56 = hiveContext.sql(m56_sql+" from "+table_user_info_201606+" as m6 left join "+table_user_info_201605+" as m5 on m6.uid = m5.uid ")
df_month56.registerTempTable(tmp_table_m56)

##56月与4月的表关联查询。
# m456_sql_part1 = "select m4.uid, m56.date, "  #暂时先不加日期。
m456_sql_part1 = "select m4.uid, "
m456_sql_part2 = ""
for i in range(0,25):
    m456_sql_part2 += " m56.m6_"+list_col[i] +", m56.m5_"+list_col[i] +", m4."+list_col[i] +" as m4_"+list_col[i] +", m56.m6_m5_"+list_col[i] +"_avg, m56.m6_m5_"+list_col[i] +"_sum , "
    m456_sql_part2 += " m56.m6_m5_"+list_col[i] +"_diff, m56.m6_m5_"+list_col[i] +"_rate, (m56.m6_m5_"+list_col[i] +"_sum + m4."+list_col[i] +")/3 as m6_m5_m4_"+list_col[i] +"_avg, "
    m456_sql_part2 += " (m56.m6_m5_"+list_col[i] +"_sum + m4."+list_col[i] +") as m6_m5_m4_"+list_col[i] +"_sum, (m56.m6_"+list_col[i] +" - m4."+list_col[i] +") as m6_m4_"+list_col[i] +"_diff, "
    m456_sql_part2 += " (case when m4."+list_col[i] +" = 0.0 then 0.0 else (m56.m6_"+list_col[i] +"/m4."+list_col[i] +") end ) as m6_m4_"+list_col[i] +"_rate,"
m456_sql_part2 = m456_sql_part2[0:-1]
m456_sql = m456_sql_part1+m456_sql_part2
df_month456 = hiveContext.sql(m456_sql+" from "+tmp_table_m56+ " as m56 left join "+table_user_info_201604+" as m4 on m56.uid = m4.uid")
# df_month456.show(5000,False)
df_month456.registerTempTable(tmp_table_m456)
##给df_month456加上新的列，表示7月是否离网。 从table_loss_201607和tmp_table_m456两个临时表进行获取。 case when cx_flag = true then 1 else 0 end
df_for_train_pre = hiveContext.sql("select m456.*, (case when loss7.leave = 1.0 then 1.0 else 0.0 end) as leave from "+tmp_table_m456+" as m456 left join "+table_loss_201607+" as loss7 on m456.uid = loss7.uid")
# df_for_train_pre.filter(("leave = 1.0")).show(1000,False)
# df_for_train_pre.printSchema()
##将dataframe中的各个列转化成label和feature列。(除去UID和日期，日期将来再用)
tmp_train_pre_rdd = df_for_train_pre.select("*").rdd
tmp_train_pre_rdd_2 = tmp_train_pre_rdd.map(lambda x:(x[1:-1],x[-1]))
tmp_train_pre_rdd_3 = tmp_train_pre_rdd_2.map(lambda p:Row(label=p[1],features=Vectors.dense(p[0])))
df_for_train_alpha = hiveContext.createDataFrame(tmp_train_pre_rdd_3,samplingRatio=0.005)
# df_for_train_alpha.show(9,False)
# df_for_train_alpha.printSchema()
# df_for_train_alpha.dtypes
##对离网用户和非离网用户进行采样处理，使正负样本数量相同。
##目前1类别（离网用户）太少，0类别（非离网用户）太多。如果要对0类别抽样，可能会造成欠拟合；如果让1类重复出现，可能会造成过拟合。
##由于0类别多的没变了，所以目前我们就从0类别中抽样出来，使0类别的数量降低到和1类别的数量相同。
##通过统计我们得知，正常用户总数乘以0.00828449，得到的数量等于离网用户的数量。
df_for_train = df_for_train_alpha.sampleBy("label", fractions={0: 0.00828, 1.0: 1})
##############以上4,5,6月用户信息进行特征处理##############



#################################################################################################################################################
#################################################################################################################################################
##上面部分是生成训练数据，接下来一部分是训练模型。
##上面部分是生成训练数据，接下来一部分是训练模型。
##上面部分是生成训练数据，接下来一部分是训练模型。
#################################################################################################################################################
#################################################################################################################################################
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed_label")
stringIndexermodel = stringIndexer.fit(df_for_train)
df_for_train_indexed = stringIndexermodel.transform(df_for_train)
rf =RandomForestClassifier(featuresCol="features", labelCol="indexed_label", predictionCol="prediction", probabilityCol="probability")
##rf_model是利用4,5,6月用户数据做features，用7月是否离网做label，使用随机森林进行训练获得的模型。
rf_model = rf.fit(df_for_train_indexed)
# print(type(rf_model))
#################################################################################################################################################
#################################################################################################################################################
##上面部分是训练模型，接下来一部分是生成测试数据及验证模型。
##上面部分是训练模型，接下来一部分是生成测试数据及验证模型。
##上面部分是训练模型，接下来一部分是生成测试数据及验证模型。
#################################################################################################################################################
#################################################################################################################################################

##############以下读取8月离网用户##############
rdd_loss_201608 = sc.textFile(loss_201608)
rdd_loss_201608 = rdd_loss_201608.map(lambda line:line.split("|"))
rdd_loss_201608_2 = rdd_loss_201608.map(lambda p : Row(uid=p[1],leave=1.0))##1表示离网
df_loss_201608 = hiveContext.createDataFrame(rdd_loss_201608_2)
df_loss_201608.registerTempTable(table_loss_201608)
##############以上读取8月离网用户##############

##############以下读取7月用户信息##############
rdd_user_info_month7 = sc.textFile(user_history_info_201607)
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
splits = rdd_user_info_month7.randomSplit({0.9, 0.1})
##测试阶段，对7月用户信息进行数据切分。正式工程化以后，切分数据的逻辑必须除去。
rdd_user_info_month7 = splits[1]

rdd_month7_2 = rdd_user_info_month7.map(lambda line:line.split("|"))
##revenue:出账收入, data:总流量, wlan:wlan流量, two_g:2g总流量 , three_g: 3g总流量,four_g:4g总流量  ,ncall: 主叫通话次数, tcall:主叫通话时长  , ncalled:被叫通话次数
##   , tcalled:被叫通话时长 , ncall_loc:本地主叫通话次数 ,ncalled_loc:本地被叫通话次数 ,tcall_loc:本地主叫通话时长 ,tcalled_loc:本地被叫通话时长 ,
## tcall_roam:漫游主叫通话时长  ,tcalled_roam:漫游被叫通话时长 ,ncall_roam:漫游主叫通话次数 ,ncalled_roam:漫游被叫通话次数
## nconv_roam_p:省内漫游通话次数 ,nconv_roam_c:国内漫游通话次数 ,nconv_roam_i:国际漫游通话次数 ,tconv_roam_p:省内漫游通话时长 ,tconv_roam_c: 国内漫游通话时长 ,tconv_roam_i:国际漫游通话时长 ,
## nmsgup:上行短信条数 ,nmsgdn:下行短信条数
rdd_month7_3 = rdd_month7_2.map(lambda p : Row(uid=p[0],date=p[4]
                                                , revenue=float('0' if (p[5] is None or p[5]=='') else p[5]), data=float('0' if (p[6] is None or p[6]=='') else p[6])
                                               , wlan=float('0' if (p[7] is None or p[7]=='') else p[7]), two_g=float('0' if (p[8] is None or p[8]=='') else p[8])
                                               , three_g=float('0' if (p[9] is None or p[9]=='') else p[9]), four_g=float('0' if (p[10] is None or p[10]=='') else p[10])
                                                , ncall=float('0' if (p[11] is None or p[11]=='') else p[11]), tcall=float('0' if (p[12] is None or p[12]=='') else p[12])
                                               , ncalled=float('0' if (p[13] is None or p[13]=='') else p[13]), tcalled=float('0' if (p[14] is None or p[14]=='') else p[14])
                                               , ncall_loc=float('0' if (p[15] is None or p[15]=='') else p[15]), ncalled_loc=float('0' if (p[16] is None or p[16]=='') else p[16])
                                               , tcall_loc=float('0' if (p[17] is None or p[17]=='') else p[17]), tcalled_loc=float('0' if (p[18] is None or p[18]=='') else p[18])
                                               ,tcall_roam=float('0' if (p[19] is None or p[19]=='') else p[19]), tcalled_roam=float('0' if (p[20] is None or p[20]=='') else p[20])
                                               , ncall_roam=float('0' if (p[21] is None or p[21]=='') else p[21]), ncalled_roam=float('0' if (p[22] is None or p[22]=='') else p[22])
                                               , nconv_roam_p=float('0' if (p[23] is None or p[23]=='') else p[23]), nconv_roam_c=float('0' if (p[24] is None or p[24]=='') else p[24])
                                               , nconv_roam_i=float('0' if (p[25] is None or p[25]=='') else p[25]) , tconv_roam_p=float('0' if (p[26] is None or p[26]=='') else p[26])
                                              , tconv_roam_c=float('0' if (p[27] is None or p[27]=='') else p[27]), tconv_roam_i=float('0' if (p[28] is None or p[28]=='') else p[28])
                                               , nmsgup=float('0' if (p[29] is None or p[29]=='') else p[29]), nmsgdn=float('0' if (p[30] is None or p[30]=='') else p[30])))
df_month07_1 = hiveContext.createDataFrame(rdd_month7_3)
print("###################################  month7  ###################################################")
# df_month07_1.show(19,False)
# df_month07_1.printSchema()
df_month07_1.registerTempTable(table_user_info_201607)
##############以上读取7月用户信息##############

##############以下利用8月离网用户uid和5,6,7月用户信息进行特征处理##############
##6,7月的表进行关联查询。（注意，不可以进行三表关联查询，笛卡尔积太大）
list_col = ["revenue","data","wlan","two_g","three_g","four_g","ncall","tcall","ncalled","tcalled","ncall_loc","ncalled_loc","tcall_loc","tcalled_loc",
            "tcall_roam","tcalled_roam","ncall_roam","ncalled_roam","nconv_roam_p","nconv_roam_c","nconv_roam_i","tconv_roam_p","tconv_roam_c","tconv_roam_i",
            "nmsgup","nmsgdn"]
# m67_sql_part1 = "select m7.uid, m7.date, "  #暂时先不加日期。
m67_sql_part1 = "select m7.uid, "
m67_sql_part2 = ""
for i in range(0,25):
    m67_sql_part2 += "m7."+list_col[i] +" as m7_"+list_col[i] +", m6."+list_col[i] +" as m6_"+list_col[i] +", (m7."+list_col[i] +"+m6."+list_col[i] +")/2 as  m7_m6_"+list_col[i] +"_avg, "
    m67_sql_part2 += "(m7."+list_col[i] +"+m6."+list_col[i] +") as m7_m6_"+list_col[i] +"_sum, (m7."+list_col[i] +"-m6."+list_col[i] +") as m7_m6_"+list_col[i] +"_diff, "
    m67_sql_part2 += "(case when m6."+list_col[i] +" = 0.0 then 0.0 else  (m7."+list_col[i] +"/m6."+list_col[i] +") end) as m7_m6_"+list_col[i] +"_rate,"
m67_sql_part2 = m67_sql_part2[0:-1]
m67_sql = m67_sql_part1+m67_sql_part2
df_month67 = hiveContext.sql(m67_sql+" from "+table_user_info_201607+" as m7 left join "+table_user_info_201606+" as m6 on m7.uid=m6.uid")
df_month67.registerTempTable(tmp_table_m67)

##67月与5月的表关联查询。
# m567_sql_part1 = "select m67.uid, m67.date, "  #暂时先不加日期。
m567_sql_part1 = "select m67.uid, "
m567_sql_part2 = ""
for i in range(0,25):
    m567_sql_part2 += " m67.m7_"+list_col[i] +", m67.m6_"+list_col[i] +", m5."+list_col[i] +" as m5_"+list_col[i] +", m67.m7_m6_"+list_col[i] +"_avg, m67.m7_m6_"+list_col[i] +"_sum , "
    m567_sql_part2 += " m67.m7_m6_"+list_col[i] +"_diff, m67.m7_m6_"+list_col[i] +"_rate, (m67.m7_m6_"+list_col[i] +"_sum + m5."+list_col[i] +")/3 as m7_m6_m5_"+list_col[i] +"_avg, "
    m567_sql_part2 += " (m67.m7_m6_"+list_col[i] +"_sum + m5."+list_col[i] +") as m7_m6_m5_"+list_col[i] +"_sum, (m67.m7_"+list_col[i] +" - m5."+list_col[i] +") as m7_m5_"+list_col[i] +"_diff, "
    m567_sql_part2 += " (case when m5."+list_col[i] +" = 0.0 then 0.0 else (m67.m7_"+list_col[i] +"/m5."+list_col[i] +") end ) as m7_m5_"+list_col[i] +"_rate,"
m567_sql_part2 = m567_sql_part2[0:-1]
m567_sql = m567_sql_part1+m567_sql_part2
df_month567 = hiveContext.sql(m567_sql+" from "+tmp_table_m67+ " as m67 left join "+table_user_info_201605+" as m5 on m67.uid=m5.uid")
df_month567.registerTempTable(tmp_table_m567)
##给df_month567加上新的列，表示8月是否离网。 从table_loss_201608和tmp_table_m567两个临时表进行获取。
df_for_test_pre = hiveContext.sql("select m567.*, (case when loss8.leave = 1.0 then 1.0 else 0.0 end) as leave from "+tmp_table_m567+" as m567 left join "+table_loss_201608+" as loss8 on m567.uid = loss8.uid")
# df_for_test_pre.show(11,False)
# df_for_test_pre.printSchema()
##############以上利用8月离网用户uid和5,6,7月用户信息进行特征处理##############

##以下将dataframe中的各个列转化成label和feature列。(除去UID和日期，日期将来再用)
tmp_test_pre_rdd = df_for_test_pre.select("*").rdd
tmp_test_pre_rdd_2 = tmp_test_pre_rdd.map(lambda x:(x[1:-1],x[-1],x[0]))
tmp_test_pre_rdd_3 = tmp_test_pre_rdd_2.map(lambda p:Row(label=p[1],features=Vectors.dense(p[0]),uid=p[2]))
df_for_test = hiveContext.createDataFrame(tmp_test_pre_rdd_3,samplingRatio=0.005)
# df_for_test.show(9,False)
# df_for_test.printSchema()
# df_for_test.dtypes
##以上将dataframe中的各个列转化成label和feature列。(除去UID和日期，日期将来再用)

##############以下进行预测及验证准确率##############
df_for_test_indexed = stringIndexermodel.transform(df_for_test)
df_result = rf_model.transform(df_for_test_indexed)
# print(type(df_result))
df_result.select("uid","indexed_label","prediction").filter(("indexed_label = 1.0")).show(3000,False)
df_result.printSchema()
evaluator_precision = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="precision")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="f1")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="recall")
accuracy = evaluator_precision.evaluate(df_result)
f1 = evaluator_f1.evaluate(df_result)
recall = evaluator_recall.evaluate(df_result)
print("accuracy ,recall and f1 : %g, %g, %g"%(accuracy, recall, f1))
##############以上进行预测及验证准确率##############