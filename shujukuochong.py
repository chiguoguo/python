# -*- coding:UTF-8 -*-
from __future__ import division
import pandas as pd
pd.set_option('display.width', 320)
pd.set_option('display.max_colwidth', -1)
pd.set_option('float_format','{:20,.2f}'.format)
from numpy import random
import matplotlib.pyplot as plt
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
import time,datetime
import numpy as np

location_source = '/home/hadoop/code/hefei/huanji/heze.csv'
location_pre1_month = '/home/hadoop/code/hefei/huanji/temp_liuxw_0705_08.txt'
location_pre2_month = '/home/hadoop/code/hefei/huanji/temp_liuxw_0705_07.txt'
location_pre3_month = '/home/hadoop/code/hefei/huanji/temp_liuxw_0705_06.txt'
target_file_path='/home/hadoop/code/hefei/huanji/new_heze.csv'
df_source = pd.read_csv(location_source)

location_change_chain = '/home/hadoop/code/hefei/huanji/change_chain'
result=[]
f = open(location_change_chain)             # 返回一个文件对象
line = f.readline()           # 调用文件的 readline()方法
while line:
    row=[]
    line = line[1:-2].replace("'", '')
    splits = line.split(',')
    #获得最后一次换机时间，上一个手机型号
    cur_chain = splits[1:]
    if len(splits) > 2:#说明换过手机
        first_ele_of_chain = cur_chain[0]
        date_last_change_phone = first_ele_of_chain.split('|')[-1]
        last_phone_model = first_ele_of_chain.split('|')[1]
        change_times = len(cur_chain)-1
        last_ele_of_chain = cur_chain[-1]
        str_start_date = last_ele_of_chain.split('|')[-1]
        start_date = time.mktime(time.strptime(str_start_date,"%y%m%d"))
        end_date = time.mktime(time.strptime(date_last_change_phone,"%y%m%d"))
        change_phone_period=(end_date - start_date)/(change_times*86400)##平均换机周期，单位是天。
    else:
        chain = splits[1]
        date_last_change_phone = chain.split('|')[-1]
        last_phone_model = chain.split('|')[1]
        ##没有换过手机，则换机周期缺省默认为0天。
        change_phone_period = 0.0
    row.append(splits[0][1:])#手机号
    row.append(date_last_change_phone)#最后一次换机时间
    row.append(last_phone_model)#上一个手机型号
    row.append(change_phone_period)#平均换机周期
    result.append(row)
    line = f.readline()
f.close()
df_change_chain = pd.DataFrame(result)
df_change_chain.columns = ['phone_number','date_last_change_phone','last_phone_model','change_phone_period']

# df_change_chain.head(10)
# df_change_chain.to_csv('/home/hadoop/code/hefei/huanji/xx.csv',encoding='utf-8')

df_change_chain['phone_number'] = df_change_chain['phone_number'].astype(np.int64)

##使用8月及8月以前的数据预测未来的换机情况的情况下,pre1_month代表8月。
##这里df_info_pre1_month存储了用户8月的出账收入及流量信息。
df_pre1_month = pd.read_csv(location_pre1_month, usecols=[0, 5, 6], names=['uid','czsr_pre1','data_pre1'], sep='|', header=None, dtype={'uid':np.int64},encoding='utf-8')
# df_pre1_month.head(10)

##使用8月及8月以前的数据预测未来的换机情况的情况下,pre2_month代表7月。
##这里df_info_pre2_month存储了用户7月的出账收入及流量信息。
df_pre2_month = pd.read_csv(location_pre2_month, usecols=[0, 5, 6], names=['uid','czsr_pre2','data_pre2'], sep='|', header=None, dtype={'uid':np.int64},encoding='utf-8')
# df_pre2_month.head(10)

##使用8月及8月以前的数据预测未来的换机情况的情况下,pre3_month代表6月。
##这里df_info_pre3_month存储了用户6月的出账收入及流量信息。
df_pre3_month = pd.read_csv(location_pre3_month, usecols=[0, 5, 6], names=['uid','czsr_pre3','data_pre3'], sep='|', header=None, dtype={'uid':np.int64},encoding='utf-8')
# df_pre3_month.head(10)

df_pre1_pre2_left_join = pd.merge(df_pre1_month, df_pre2_month, how='left', on='uid')
df_pre1_pre2_pre3_left_join = pd.merge(df_pre1_pre2_left_join, df_pre3_month, how='left', on='uid')

df_pre1_pre2_pre3_left_join['sum_liuliang_of_pre_3_months']=df_pre1_pre2_pre3_left_join['data_pre1']+df_pre1_pre2_pre3_left_join['data_pre2']+df_pre1_pre2_pre3_left_join['data_pre3']
df_pre1_pre2_pre3_left_join['czsr_of_pre_3_months']=df_pre1_pre2_pre3_left_join['czsr_pre1'] - df_pre1_pre2_pre3_left_join['czsr_pre3']
df_info_pre_3_months = df_pre1_pre2_pre3_left_join[['uid','sum_liuliang_of_pre_3_months','czsr_of_pre_3_months']]
df_info_pre_3_months['uid'] = df_info_pre_3_months['uid'].astype(np.int64)

##读取源文件，即需要被补充的文件
df_source = pd.read_csv(location_source, dtype={'uid':np.int64})

f_source_2 = pd.merge(df_source, df_info_pre_3_months,on='uid', how='left')

##f_source_2还要和change_chain中得出来的df_change_chain进行left join，通过手机号。
f_source_2['phone_number'] = f_source_2['phone_number'].astype(np.int64)

f_source_3 = pd.merge(f_source_2, df_change_chain, how='left', on='phone_number')

f_source_3.to_csv(target_file_path,encoding='utf-8')

