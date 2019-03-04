import os
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random


pd.set_option('max_colwidth',200)
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
# %matplotlib inline
le = LabelEncoder()

# data root dir
dataroot = 'F:/数据集/1207甜橙金融/data/'
cacheRoot = 'F:/数据集处理/甜橙/'

# 读取合并 等预处理
# -------预处理函数

def dataRank(data,rankdate):
    data['dtimeRANK'] = data.groupby('UID')[rankdate].rank(ascending=True)
    return data

def dayandtime2date(dayint,timestr):
    start_datestr = "2018-10-31"
    start_date = datetime.datetime.strptime(start_datestr, '%Y-%m-%d')
    timestrlist = [int(x) for x in timestr.split(":")]
    del start_datestr
    datadate = start_date + datetime.timedelta(days=dayint)
    return datetime.datetime.combine(datadate, datetime.time(*timestrlist))
def dif_to_sec(x):
    try:
        d = abs(x).seconds
    except Exception as e:
        d = np.nan
    return d
def get_hour(x):
    try:
        h = x.hour
    except Exception as e:
        h = np.nan
    return h


def diffWindow600Secd(train_oper,justTran):
    '''
    > datadate: combined day and time base on 2018/10/31 tran:1-30 test 31:61
    > dtimeRANK:useless replace by sort inplace
    > time_group_cumcount: use cumcount generate user oper in window of 600 seconds time diff
    > useful : time_diff / time_group_cumcount
    '''
    
    train_oper["datadate"] = train_oper.apply(lambda x: dayandtime2date(x["megday"],x["time"]),axis=1)
#     train_oper = dataRank(train_oper,"datadate")
    train_oper.sort_values(['UID', 'datadate'], inplace=True)
    train_oper["time_diff"] = train_oper.groupby("UID")["datadate"].diff(periods=-1)
    train_oper["time_diff_secd"] = train_oper["time_diff"].map(dif_to_sec)
    train_oper["time_diff_secd"]= train_oper["time_diff_secd"].fillna(99999)
    if justTran:
        # trans_amt/bal
        train_oper["trans_amt_dif"] = train_oper.groupby("UID")["trans_amt"].diff(periods=1)
        train_oper["trans_amt_dif"]= train_oper["trans_amt_dif"].fillna(0)
        train_oper["bal_dif"] = train_oper.groupby("UID")["bal"].diff(periods=1)
        train_oper["bal_dif"]= train_oper["bal_dif"].fillna(0)
        
    train_oper["time_group_tmp"] = train_oper["time_diff_secd"].map(lambda x: 1 if x>600 else np.nan)
    train_oper["time_group_tmp_cumcount"] = train_oper.groupby(["UID","time_group_tmp"]).cumcount()
    train_oper["time_group_tmp_cumcount"] = train_oper["time_group_tmp_cumcount"] *  train_oper["time_group_tmp"] 
    train_oper["time_group"] = train_oper["time_group_tmp_cumcount"].fillna(method='bfill')
    train_oper = train_oper.drop(["time_group_tmp" , "time_group_tmp_cumcount","time_diff"],axis=1)
    return train_oper


# 读取
trainFiles  = {"transaction":'transaction_train_new.csv', "operation" :'operation_train_new.csv', "label" :'tag_train_new.csv'} 
validFiles = {"transaction":'transaction_round1_new.csv', "operation" :'operation_round1_new.csv',"label":"提交样例.csv"}
testFiles  = {"transaction":'test_transaction_round2.csv', "operation" :'test_operation_round2.csv',"submit":"submit_example.csv"}
# data
# # # train operation
train_oper = pd.read_csv(open(dataroot + trainFiles['operation'],encoding="utf8"))
# # train transaction
train_transac = pd.read_csv(open(dataroot + trainFiles['transaction'],encoding="utf8"))


test_oper_mid = pd.read_csv(open(dataroot + validFiles["operation"],encoding="utf8"))
# # test transaction
test_transac_mid = pd.read_csv(open(dataroot + validFiles["transaction"],encoding="utf8"))

test_oper  = pd.read_csv(open(dataroot + testFiles["operation"],encoding="utf8"))
# # test transaction
test_transac = pd.read_csv(open(dataroot + testFiles["transaction"],encoding="utf8"))


# # # id
trainTag = pd.read_csv(open(dataroot + validFiles['label'],encoding="utf8"))
submission = pd.read_csv(open(dataroot + testFiles['submit']))
# LonAndLat = pd.read_csv("D:/COMPETITIONS/TianChen/GEO/GeoDetails.csv")


# 合并
tmp_trn_oper = train_oper.copy()
tmp_trn_tran = train_transac.copy()

tmp_trn_opmi = test_oper_mid.copy()
tmp_trn_trmi = test_transac_mid.copy()

tmp_tst_oper = test_oper.copy()
tmp_tst_tran = test_transac.copy()

#add day connect

tmp_trn_oper["megday"] = tmp_trn_oper["day"]
tmp_trn_tran["megday"] = tmp_trn_tran["day"]

tmp_trn_opmi["megday"] = tmp_trn_opmi["day"] + 31
tmp_trn_trmi["megday"] = tmp_trn_trmi["day"] + 31 

tmp_tst_oper["megday"] = tmp_tst_oper["day"] + 30 + 31
tmp_tst_tran["megday"] = tmp_tst_tran["day"] + 30 + 31



use_oper = pd.concat([ tmp_trn_oper, tmp_trn_opmi,tmp_tst_oper])
use_tran = pd.concat([ tmp_trn_tran, tmp_trn_trmi, tmp_tst_tran ])

trainLabel = pd.read_csv(trainFiles['label'])
submiLabel = pd.read_csv(testFiles['submit'])
data = pd.concat( [ trainLabel ,  submiLabel] )
del tmp_trn_oper,tmp_trn_tran,tmp_trn_opmi,tmp_trn_trmi,tmp_tst_oper,tmp_tst_tran,train_oper,train_transac,test_oper_mid,test_transac_mid,test_oper,test_transac

 # 预处理2

tmp_oper = use_oper.copy()
tmp_oper["is_oper"] = 1
use_all = pd.concat([tmp_oper,use_tran ])
use_all["is_oper"] = use_all["is_oper"].fillna(0)
use_all["ip"] = use_all["ip1"].fillna(use_all["ip2"])
del tmp_oper
use_all = diffWindow600Secd(use_all, False)
use_all["ip"] = use_all["ip1"].fillna(use_all["ip2"])
use_all["hour"] = use_all.datadate.map(get_hour)
use_all["device_code"] = use_all["device_code1"].fillna(use_all["device_code2"])
use_all["device_code"] = use_all["device_code"].fillna(use_all["device_code3"])

use_oper = diffWindow600Secd(use_oper, False)
use_oper["ip"] = use_oper["ip1"].fillna(use_oper["ip2"])
use_oper["hour"] = use_oper.datadate.map(get_hour)
use_oper.wifi = use_oper.wifi.fillna("Phonetraffic")

use_tran = diffWindow600Secd(use_tran, True)
use_tran["hour"] = use_tran.datadate.map(get_hour)
use_tran["bal_increase"] = use_tran["bal_dif"].map(lambda x: x if x >0 else 0)
use_tran["bal_decrease"] = use_tran["bal_dif"].map(lambda x: x if x <0 else 0)
use_tran["bal_difabs"] = use_tran["bal_dif"].map(abs)
use_tran["trans_amt_increase"] = use_tran["trans_amt_dif"].map(lambda x: x if x >0 else 0)
use_tran["trans_amt_decrease"] = use_tran["trans_amt_dif"].map(lambda x: x if x <0 else 0)
use_tran["trans_amt_difabs"] = use_tran["trans_amt_dif"].map(abs)
use_tran["bal_decrease2amt"] = use_tran["bal_decrease"]/use_tran["trans_amt"]
use_tran["bal_decrease2amt"] = use_tran["bal_decrease2amt"].map(abs)
use_tran["device_code"] = use_tran["device_code1"].fillna(use_tran["device_code2"])
use_tran["device_code"] = use_tran["device_code"].fillna(use_tran["device_code3"])

for c in ["os","wifi"]:
    use_all[c] = use_all.groupby("UID")[c].apply(lambda x: x.ffill())

use_all["version_before_tran"] = use_all.groupby("UID")["version"].apply(lambda x: x.ffill()) 
use_all["version_after_tran"] = use_all.groupby("UID")["version"].apply(lambda x: x.bfill()) 
use_all["mode_before_tran"] = use_all.groupby("UID")["mode"].apply(lambda x: x.ffill()) 
use_all["mode_after_tran"] = use_all.groupby("UID")["mode"].apply(lambda x: x.bfill()) 

# use_all.head()


import gc
gc.collect()

# 特征工程
# block1
# block 1 use_all
block1 = pd.DataFrame()
block1["UID"] = data["UID"]

gp1 = use_all.groupby(["UID",'time_group'])['version'].agg({ "version_short_time":"nunique" }).reset_index()
gp = gp1.groupby("UID")["version_short_time"].agg({ "version_short_time_max":"max",
                                                  "version_short_time_mean":"mean"}).reset_index()
block1 = block1.merge(gp,on = ["UID"],how = "left")
del gp,gp1
gp1 = use_all[use_all.device2.isnull()].groupby("UID")["mode"].agg({"device2null_most_mode":lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan ,
                                "device2null_most_mode_cnt":lambda x:x.value_counts().values[0] if len(x.value_counts())>1 else np.nan ,
                                "device2null_scnd_mode":lambda x: x.value_counts().index[1] if len(x.value_counts())>1 else np.nan ,
                                "device2null_scnd_mode_cnt":lambda x:x.value_counts().values[1] if len(x.value_counts())>1 else np.nan 
                               }).reset_index()
gp1["device2null_scnd2most_mode_ratio"] = gp1["device2null_scnd_mode_cnt"]/gp1["device2null_most_mode_cnt"]
to_encode = list( set(gp1.device2null_most_mode.map(str).unique().tolist() + gp1.device2null_scnd_mode.map(str).unique().tolist()) )
le.fit(to_encode )
gp1.device2null_most_mode = le.transform(gp1.device2null_most_mode.map(str) )
gp1.device2null_scnd_mode = le.transform(gp1.device2null_scnd_mode.map(str))
block1 = block1.merge(gp1,on = ["UID"],how = "left")
del gp1



# block 2

import gc
# device2/geo_code/  os/version/success/wifi/  channel/day/hour/time_diff_secd
# count version_cnt in time_froup 表征刷机（版本更替频繁 短时间） 最好 对version 做 befeore after操作
# 避免刷机影响 求 max version in time_group 后再求最小版本
# 刷机pattern 2 一组时间内，要么交易，要么操作 deveice2 为空
# 统计 mode对 device2 各种操作缺失，或者缺失下的 mode 第一是什么第二是什么 第二/第一 比例
# 统计device_code1空值比
# device2 Tag1 的 IPTHON 5S很多，0的基本没有5S，直接对用户手机device2 most值lable encode

# block 2 use_all
block2 = pd.DataFrame()
block2["UID"] = data["UID"]




# use_all
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['UID','device2','channel'], 'select': 'day', 'agg': 'var'},
    {'groupby': ['UID','ip','channel'], 'select': 'day', 'agg': 'var'},
    {'groupby': ['UID','device2','ip'], 'select': 'day', 'agg': 'var'},
    {'groupby': ['UID','version','channel'], 'select': 'day', 'agg': 'var'},
    
    {'groupby': ['UID','ip','version'], 'select': 'hour', 'agg': 'var'},
    {'groupby': ['UID','device2','ip'], 'select': 'hour', 'agg': 'var'},
    
    # Variance in hour, for ip-app-os
    {'groupby': ['UID','device2','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['UID','day','channel'], 'select': 'hour', 'agg': 'var'},
    
    # Count, for ip-day-hour
    {'groupby': ['UID','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['UID', 'device2'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['UID', 'device2', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['UID','device2','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['UID','device2','channel'], 'select': 'hour', 'agg': 'mean'}, 
  
    # Count, for UID-day-hour
    {'groupby': ['UID','day','hour'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['UID', 'device2'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['UID', 'device2', 'os'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['UID','device2','day','hour'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['UID','device2','channel'], "query":'time_diff_secd < 5 ',"agg_name": "var_ude5",'select': 'hour', 'agg': 'mean'}, 

    {'groupby': ['UID','day','hour'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'is_oper', 'agg': 'mean'},
    # Count, for UID-device2-day-hour is_oper
    {'groupby': ['UID', 'device2'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'is_oper', 'agg': 'mean'},        
    {'groupby': ['UID', 'device2', 'os'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'is_oper', 'agg': 'mean'},
    {'groupby': ['UID','device2','day','hour'],"query":'time_diff_secd < 5 ',"agg_name": "var_ude5", 'select': 'is_oper', 'agg': 'mean'},

    # Count, time_diff_secd < 10 for UID-device2-day-hour-os time_diff_secd
    {'groupby': ['UID','day','hour'],"query":'time_diff_secd < 10 ',"agg_name": "var_ude10", 'select': 'time_diff_secd', 'agg': 'nunique'},
    {'groupby': ['UID', 'device2'],"query":'time_diff_secd < 10 ',"agg_name": "var_ude10", 'select': 'time_diff_secd', 'agg': 'nunique'},        
    {'groupby': ['UID', 'device2', 'os'],"query":'time_diff_secd < 10 ',"agg_name": "var_ude10", 'select': 'time_diff_secd', 'agg': 'nunique'},
    {'groupby': ['UID','device2','day','hour'],"query":'time_diff_secd <10 ',"agg_name": "var_ude10", 'select': 'time_diff_secd', 'agg': 'nunique'},
    # Count, time_diff_secd < 10 for UID-device2-day-hour-os device_code
    {'groupby': ['UID','day',    'hour'],"query":'time_diff_secd < 10 ',"agg_name": "var_ude10", 'select': 'device_code', 'agg': 'nunique'},
    {'groupby': ['UID',          'hour'],"query":'time_diff_secd < 10 ',"agg_name": "var_ude10", 'select': 'device_code', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'time_diff_secd < 10 ',"agg_name": "var_ude10", 'select': 'device_code', 'agg': 'nunique'},
    # Count, day < 10 for UID-device2-day-hour-os merchant
    {'groupby': ['UID','day',    'hour'],"query":'day < 10 ',"agg_name": "var_dayude10", 'select': 'merchant', 'agg': 'nunique'},
    {'groupby': ['UID',          'hour'],"query":'day < 10 ',"agg_name": "var_dayude10", 'select': 'merchant', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'day < 10 ',"agg_name": "var_dayude10", 'select': 'merchant', 'agg': 'nunique'},
    # Count, day < 10 for UID-device2-day-hour-os hour
    {'groupby': ['UID','day','merchant'],"query":'day < 10 ',"agg_name": "var_dayude10", 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['UID',      'merchant'],"query":'day < 10 ',"agg_name": "var_dayude10", 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'day < 10 ',"agg_name": "var_dayude10", 'select': 'hour', 'agg': 'nunique'},    
     # Count, day < 20 for UID-device2-day-hour-os merchant
    {'groupby': ['UID','day',    'hour'],"query":'day < 20 ',"agg_name": "var_dayude20", 'select': 'merchant', 'agg': 'nunique'},
    {'groupby': ['UID',          'hour'],"query":'day < 20 ',"agg_name": "var_dayude20", 'select': 'merchant', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'day < 20 ',"agg_name": "var_dayude20", 'select': 'merchant', 'agg': 'nunique'},
    # Count, day < 20 for UID-device2-day-hour-os hour
    {'groupby': ['UID','day','merchant'],"query":'day < 20 ',"agg_name": "var_dayude20", 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['UID',      'merchant'],"query":'day < 20 ',"agg_name": "var_dayude20", 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'day < 20 ',"agg_name": "var_dayude20", 'select': 'hour', 'agg': 'nunique'},    
    # Count, day < 32 for UID-device2-day-hour-os merchant
    {'groupby': ['UID','day',    'hour'],"query":'day < 32 ',"agg_name": "var_dayude32", 'select': 'merchant', 'agg': 'nunique'},
    {'groupby': ['UID',          'hour'],"query":'day < 32 ',"agg_name": "var_dayude32", 'select': 'merchant', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'day < 32 ',"agg_name": "var_dayude32", 'select': 'merchant', 'agg': 'nunique'},
     # Count, day < 32 for UID-device2-day-hour-os hour
    {'groupby': ['UID','day','merchant'],"query":'day < 32 ',"agg_name": "var_dayude32", 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['UID',      'merchant'],"query":'day < 32 ',"agg_name": "var_dayude32", 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['UID','day'           ],"query":'day < 32 ',"agg_name": "var_dayude32", 'select': 'hour', 'agg': 'nunique'},    
    
    
    # var/sum, time_diff_secd < 600/300 for UID-device2-day-ip-os time_diff_secd
    {'groupby': ['UID',    'device2'],"query":'time_diff_secd < 600 ',"agg_name": "var_ude600", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',    'device2'],"query":'time_diff_secd < 300 ',"agg_name": "var_ude300", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',    'device2'],"query":'time_diff_secd < 600 ',"agg_name": "sum_ude600", 'select': 'time_diff_secd', 'agg': 'sum'},
    {'groupby': ['UID',    'device2'],"query":'time_diff_secd < 300 ',"agg_name": "sum_ude300", 'select': 'time_diff_secd', 'agg': 'sum'},
    
    {'groupby': ['UID',        'day'],"query":'time_diff_secd < 600 ',"agg_name": "var_ude600", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',        'day'],"query":'time_diff_secd < 300 ',"agg_name": "var_ude300", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',        'day'],"query":'time_diff_secd < 600 ',"agg_name": "sum_ude600", 'select': 'time_diff_secd', 'agg': 'sum'},
    {'groupby': ['UID',        'day'],"query":'time_diff_secd < 300 ',"agg_name": "sum_ude300", 'select': 'time_diff_secd', 'agg': 'sum'},
    
    {'groupby': ['UID',         'ip'],"query":'time_diff_secd < 600 ',"agg_name": "var_ude600", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',         'ip'],"query":'time_diff_secd < 300 ',"agg_name": "var_ude300", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',         'ip'],"query":'time_diff_secd < 600 ',"agg_name": "sum_ude600", 'select': 'time_diff_secd', 'agg': 'sum'},
    {'groupby': ['UID',         'ip'],"query":'time_diff_secd < 300 ',"agg_name": "sum_ude300", 'select': 'time_diff_secd', 'agg': 'sum'},
    
    {'groupby': ['UID',         'os'],"query":'time_diff_secd < 600 ',"agg_name": "var_ude600", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',         'os'],"query":'time_diff_secd < 300 ',"agg_name": "var_ude300", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',         'os'],"query":'time_diff_secd < 600 ',"agg_name": "sum_ude600", 'select': 'time_diff_secd', 'agg': 'sum'},
    {'groupby': ['UID',         'os'],"query":'time_diff_secd < 300 ',"agg_name": "sum_ude300", 'select': 'time_diff_secd', 'agg': 'sum'},
    
    {'groupby': ['UID',       'wifi'],"query":'time_diff_secd < 600 ',"agg_name": "var_ude600", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',       'wifi'],"query":'time_diff_secd < 300 ',"agg_name": "var_ude300", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID',       'wifi'],"query":'time_diff_secd < 600 ',"agg_name": "sum_ude600", 'select': 'time_diff_secd', 'agg': 'sum'},
    {'groupby': ['UID',       'wifi'],"query":'time_diff_secd < 300 ',"agg_name": "sum_ude300", 'select': 'time_diff_secd', 'agg': 'sum'},    
    
    {'groupby': ['UID','market_code'],"query":'time_diff_secd < 600 ',"agg_name": "var_ude600", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID','market_code'],"query":'time_diff_secd < 300 ',"agg_name": "var_ude300", 'select': 'time_diff_secd', 'agg': 'var'},
    {'groupby': ['UID','market_code'],"query":'time_diff_secd < 600 ',"agg_name": "sum_ude600", 'select': 'time_diff_secd', 'agg': 'sum'},
    {'groupby': ['UID','market_code'],"query":'time_diff_secd < 300 ',"agg_name": "sum_ude300", 'select': 'time_diff_secd', 'agg': 'sum'},        
   
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    celldata = pd.DataFrame()
    celldata["UID"] = data["UID"]
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    if "query" in spec:
        d_gp = use_all.query(spec['query'])[all_features]
    else:
        d_gp = use_all[all_features]
    gp = d_gp. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature})
    
    
    gpc = gp.groupby('UID')[new_feature].agg({new_feature + "_max" :"max",
                                            new_feature + "_min" :"min",
                                            new_feature + "_mean" :"mean"}).reset_index()
    
    block2 = block2.merge(gpc,on = "UID" ,how = 'left'       )
    
    del gp,gpc,d_gp
    gc.collect()



# block 3

# block 3 use_tran
block3 = pd.DataFrame()
block3["UID"] = data["UID"]




# use_all
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features # for bal and trans_amt diff
    #########################    
    # Variance in day,hour,ip,channel,merchant,time_group, conditon time_diff_secd/day
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"bal_increase >0 ",'select': 'bal_increase', 'agg': "nunique"},


{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"bal_difabs >0 ",'select': 'bal_difabs', 'agg': "nunique"},

{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"trans_amt_increase >0 ",'select': 'trans_amt_increase', 'agg': "nunique"},


{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'mean'},
{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','day'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': "nunique"},

{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','hour'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': "nunique"},

{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': 'mean'},
{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs',"agg_name": "mod", 'agg': lambda x:x.value_counts().index[0] if len(x.value_counts())>1 else np.nan},
{'groupby': ['UID','channel'], 'query':"trans_amt_difabs >0 ",'select': 'trans_amt_difabs', 'agg': "nunique"},
{'groupby': ['UID','day'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'var'},
{'groupby': ['UID','day'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'mean'},
{'groupby': ['UID','hour'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'var'},
{'groupby': ['UID','hour'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'mean'},

{'groupby': ['UID','channel'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'var'},
{'groupby': ['UID','channel'], 'query':"bal_decrease2amt >0 ",'select': 'bal_decrease2amt', 'agg': 'mean'},

]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    celldata = pd.DataFrame()
    celldata["UID"] = data["UID"]
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    if "query" in spec:
        d_gp = use_tran.query(spec['query'])[all_features]
    else:
        d_gp = use_tran[all_features]
    gp = d_gp. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature})
    
    
    gpc = gp.groupby('UID')[new_feature].agg({new_feature + "_max" :"max",
                                            new_feature + "_min" :"min",
                                            new_feature + "_mean" :"mean"}).reset_index()
    
    block3 = block3.merge(gpc,on = "UID" ,how = 'left'       )
    
        

        #     Merge back to X_total
#     if 'cumcount' == spec['agg']:
#         X_train[new_feature] = gp[0].values
#     else:
#         X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp,gpc
    gc.collect()


# 多角度下用户属性扩散后的指标，比如用户多个ip中用户量/交易量/地址数各是多少，用户在维度可统计的（交易量）占据属性本身比例
# 属性可算什么/用户是否同时可算，比率还是比值，属性下用户可算值均值，用户在属性可算值比去这个值，或者统计所有属性值的该值（可算值分布，不是属性值分布）的分布
#                |---ip1/mac1  num of users/trans/geos and USR ratio in ip1
# USR(summary) --|---ip2/mac2  num of users/trans/geos and USR ratio in ip2
#                |---ip3/mac3  num of users/trans/geos and USR ratio in ip3
# is_ratio 统计用户 可算值/属性可算值均值 可算值/属性可算值总值
gc.collect()
block4 = pd.DataFrame()
block4["UID"] = data["UID"]

dataUse = {"use_tran":use_tran,"use_oper":use_oper}
GroupOnCags = [
{'groupby': ['UID'       ],'use':"use_tran" ,'is_ratio':{'gpby':["mac1"    ,'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'       ],'use':"use_tran" ,'is_ratio':{'gpby':["geo_code",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'       ],'use':"use_tran" ,'is_ratio':{'gpby':["device_code",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'       ],'use':"use_tran" ,'is_ratio':{'gpby':["device_code1",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'       ],'use':"use_oper" ,'is_ratio':{'gpby':["mac2",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'       ],'use':"use_tran" ,'is_ratio':{'gpby':["ip1",'UID'],"slt":'day','agn':'trans','ag':"count"} },
{'groupby': ['UID'       ],'use':"use_tran" ,'is_ratio':{'gpby':["merchant",'UID'],"slt":'day','agn':'trans','ag':"count"} }
# {'groupby': ['UID'       ], 'is_summy':[ ],'select': 'UID', 'agg': 'var',},
# {'groupby': ['UID', 'day'], 'is_summy':[ ],'select': 'UID', 'agg': 'var',},
# {'groupby': ['UID', 'day'], 'is_ratio':[ ],'select': 'UID', 'agg': 'var',}
]

for spec in GroupOnCags:
    celldata = pd.DataFrame()
    celldata["UID"] = data["UID"]    
    # Name of the aggregation we're applying
#     agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
#     # Name of new feature
#     new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    # Info
#     print("Grouping by {}, and aggregating {} with {}".format(
#         spec['groupby'], spec['select'], agg_name
#     ))
    
    if "is_ratio" in spec:    
        is_ratio = spec['is_ratio']
        agg_n = is_ratio['agn'] if 'agn' in is_ratio else is_ratio['ag']
        ratio_on_name = '{}_{}'.format('_'.join(is_ratio['gpby']), agg_n)
        print("Grouping by {}, and aggregating {} with {}".format(
        is_ratio['gpby'], is_ratio['slt'], agg_n
            ))
        if "UID" in is_ratio['gpby']:
            print(is_ratio['gpby'])
            print( is_ratio['slt'])
            print(is_ratio['ag'])
            each = dataUse[spec['use']].groupby(is_ratio['gpby'])[is_ratio['slt']].agg(is_ratio['ag']).reset_index().rename(index=str, columns={is_ratio['slt']: ratio_on_name})
#             each = each.fillna(0.00001)
            total = each.groupby(is_ratio['gpby'][0])[ratio_on_name].agg({ ratio_on_name + "_mean":"mean",
                                                                           ratio_on_name + "_sum":"sum"}).reset_index()
            gp = pd.merge(each,total,on=[ is_ratio['gpby'][0]] )
            gp[ "user_ratio_on_" + ratio_on_name + "_mean" ] = gp[ratio_on_name]/gp[ratio_on_name + "_mean"]
            gp[ "user_ratio_on_" + ratio_on_name + "_sum" ] = gp[ratio_on_name]/gp[ratio_on_name + "_sum"]
            gp = gp.drop(ratio_on_name,axis =1 )
            for c in gp.columns:
                if c in is_ratio['gpby']:
                    pass
                else:
                    gpc = gp.groupby("UID")[c].agg({ c + "_maxon" +is_ratio['gpby'][0] :"max" ,
                                                     c + "_avgon" +is_ratio['gpby'][0] :"mean" ,
                                                     c + "_minon" +is_ratio['gpby'][0] :"min"}).reset_index()
                    block4 = block4.merge(gpc,on = "UID" ,how = 'left')
                    del gpc
            
            
            
            del gp,each,total
# del tmp


# 保存并合并特征
gc.collect()

for bk in [block1,block2,block3,block4]:
    data = data.merge(bk,on=["UID"],how="left")
saveName = "grouping_features.csv"
data.to_csv(cacheRoot + saveName ,encoding = "utf8", index = False)



