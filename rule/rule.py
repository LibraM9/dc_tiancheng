# -*- coding: utf-8 -*-
#@author: limeng
#@file: rule.py
#@time: 2018/12/7 20:12
"""
文件说明：找各种字段的规则，以人维度为准
"""
import pandas as pd

path = 'F:/数据集/1207甜橙金融/data/'

f = open(path+ 'operation_train_new.csv',encoding='utf8')
op_train = pd.read_csv(f)
f = open(path+ 'transaction_train_new.csv',encoding='utf8')
trans_train = pd.read_csv(f)

f = open(path+ 'test_operation_round2.csv',encoding='utf8')
op_test = pd.read_csv(f)
f = open(path+ 'test_transaction_round2.csv',encoding='utf8')
trans_test = pd.read_csv(f)

f = open(path+ 'tag_train_new.csv',encoding='utf8')
y = pd.read_csv(f)
#####bad##############################
bad = y[y.Tag == 1]
op_bad = op_train[op_train.UID.isin(bad.UID.values)]
op_bad = op_bad.sort_values(by=['UID'], ascending=(True))
trans_bad = trans_train[trans_train.UID.isin(bad.UID.values)]
trans_bad = trans_bad.sort_values(by=['UID'], ascending=(True))

# oper_bad.to_excel(path+'oper_bad.xlsx',index=None)
# trans_bad.to_excel(path+'trans_bad.xlsx',index=None)

#坏规则,人维度
trans_train = trans_train.merge(y, on='UID', how='left')
op_train = op_train.merge(y, on='UID', how='left')
bad_list = list(bad.UID.values)
def find_wrong(trans_train, feature):
    black = (trans_train.groupby([feature])['UID'].unique().apply(lambda x: len(set(x).intersection(set(bad_list)))) / trans_train.groupby([feature])['UID'].nunique()).sort_values(
        ascending=False)
    tag_count = trans_train.groupby([feature])['UID'].nunique().reset_index()
    black = black.reset_index()
    black = black.merge(tag_count, on=feature, how='left')
    black = black.sort_values(by=[ 'UID_x','UID_y'], ascending=False)
    return black

#trans#####################################################
#merchant #有用 0.95 10
black_merchant = find_wrong(trans_train, 'merchant')
rule_code = black_merchant[(black_merchant.UID_x>=0.9)&(black_merchant.UID_y>=5)].merchant.tolist()
test_rule_uid_11 = trans_test[trans_test['merchant'].isin(rule_code)].UID.unique()
#device_code1 有效
black_device_code1 = find_wrong(trans_train, 'device_code1')
rule_code = black_device_code1[(black_device_code1.UID_x>=0.97)&(black_device_code1.UID_y>=20)].device_code1.tolist()
test_rule_uid_12 = trans_test[trans_test['device_code1'].isin(rule_code)].UID.unique()
# #ip1
# black_ip1 = find_wrong(trans_train, 'ip1')
# rule_code = black_ip1[(black_ip1.UID_x>=0.97)&(black_ip1.UID_y>=20)].ip1.tolist()
# test_rule_uid_13 = trans_test[trans_test['ip1'].isin(rule_code)].UID.unique()
#acc_id1 有效
black_acc_id1 = find_wrong(trans_train, 'acc_id1')
rule_code = black_acc_id1[(black_acc_id1.UID_x>=0.97)&(black_acc_id1.UID_y>=20)].acc_id1.tolist()
test_rule_uid_13 = trans_test[trans_test['acc_id1'].isin(rule_code)].UID.unique()
#acc_id2 有效
black_acc_id2 = find_wrong(trans_train, 'acc_id2')
rule_code = black_acc_id2[(black_acc_id2.UID_x>=0.97)&(black_acc_id2.UID_y>=20)].acc_id2.tolist()
test_rule_uid_14 = trans_test[trans_test['acc_id2'].isin(rule_code)].UID.unique()
#acc_id3 有效
black_acc_id3 = find_wrong(trans_train, 'acc_id3')
rule_code = black_acc_id3[(black_acc_id3.UID_x>=0.97)&(black_acc_id3.UID_y>=20)].acc_id3.tolist()
test_rule_uid_15 = trans_test[trans_test['acc_id3'].isin(rule_code)].UID.unique()
# #geo_code
# black_geo_code = find_wrong(trans_train, 'geo_code')
# rule_code = black_geo_code[(black_geo_code.UID_x>=0.97)&(black_geo_code.UID_y>=20)].geo_code.tolist()
# test_rule_uid_16 = trans_test[trans_test['geo_code'].isin(rule_code)].UID.unique()
#oper######################################################
# #geo_code
# black_geo_code= find_wrong(op_train, 'geo_code')
# rule_code = black_geo_code[(black_geo_code.UID_x>=0.97)&(black_geo_code.UID_y>=20)].geo_code.tolist()
# test_rule_uid_21 = op_test[op_test['geo_code'].isin(rule_code)].UID.unique()
#device_code1 #有效
black_device_code1= find_wrong(op_train, 'device_code1')
rule_code = black_device_code1[(black_device_code1.UID_x>=0.97)&(black_device_code1.UID_y>=20)].device_code1.tolist()
test_rule_uid_22 = op_test[op_test['device_code1'].isin(rule_code)].UID.unique()
#mac1 有效
black_mac1= find_wrong(op_train, 'mac1')
rule_code = black_mac1[(black_mac1.UID_x>=0.97)&(black_mac1.UID_y>=20)].mac1.tolist()
test_rule_uid_23 = op_test[op_test['mac1'].isin(rule_code)].UID.unique()
# #ip1
# black_ip1= find_wrong(op_train, 'ip1')
# rule_code = black_ip1[(black_ip1.UID_x>=0.97)&(black_ip1.UID_y>=20)].ip1.tolist()
# test_rule_uid_21 = op_test[op_test['ip1'].isin(rule_code)].UID.unique()
# #ip2
# black_ip2= find_wrong(op_train, 'ip2')
# rule_code = black_ip2[(black_ip2.UID_x>=0.97)&(black_ip2.UID_y>=20)].ip2.tolist()
# test_rule_uid_21 = op_test[op_test['ip2'].isin(rule_code)].UID.unique()
#device_code3
# black_device_code3= find_wrong(op_train, 'device_code3')
# rule_code = black_device_code3[(black_device_code3.UID_x>=0.97)&(black_device_code3.UID_y>=20)].device_code3.tolist()
# test_rule_uid_21 = op_test[op_test['device_code3'].isin(rule_code)].UID.unique()
#mac2 有效
black_mac2= find_wrong(op_train, 'mac2')
rule_code = black_mac2[(black_mac2.UID_x>=0.97)&(black_mac2.UID_y>=20)].mac2.tolist()
test_rule_uid_24 = op_test[op_test['mac2'].isin(rule_code)].UID.unique()

#规则应用
outpath = 'F:/数据集/1207甜橙金融/out/'
auc = 21
f = open('F:/数据集/1207甜橙金融/out/baseline_%s.csv' % str(auc), encoding='utf8')
submit = pd.read_csv(f)

# test_rule_uid = pd.DataFrame(test_rule_uid_1)
test_rule_uid = pd.DataFrame(list(set(test_rule_uid_11).union(set(test_rule_uid_12)).union(set(test_rule_uid_13))\
                             .union(set(test_rule_uid_14)).union(set(test_rule_uid_15))\
                             .union(set(test_rule_uid_22)).union(set(test_rule_uid_23)).union(set(test_rule_uid_24))))
pred_data_rule = submit.merge(test_rule_uid, left_on='UID', right_on=0, how='left')
pred_data_rule['Tag'][(pred_data_rule[0] > 0)] = 1
pred_data_rule[['UID', 'Tag']].to_csv(outpath+'baseline_{}_badrule8.csv'.format(auc), index=False)


##########好规则##############################################
good = y[y.Tag == 0]
good_list = good.UID.tolist()
def find_good(trans_train, feature):
    white = (trans_train.groupby([feature])['UID'].unique().apply(lambda x: len(set(x).intersection(set(good_list)))) / trans_train.groupby([feature])['UID'].nunique()).sort_values(
        ascending=False)
    tag_count = trans_train.groupby([feature])['UID'].nunique().reset_index()
    white = white.reset_index()
    white = white.merge(tag_count, on=feature, how='left')
    white = white.sort_values(by=[ 'UID_x','UID_y'], ascending=False)
    return white

#trans###################
#merchant 覆盖598
white_merchant= find_good(trans_train, 'merchant')
rule_code = white_merchant[(white_merchant.UID_x>=1)&(white_merchant.UID_y>=200)].merchant.tolist()
test_rule_uid_11 = trans_test[trans_test['merchant'].isin(rule_code)].UID.unique()
#acc_id1用户交易账户号 覆盖48
white_acc_id1= find_good(trans_train, 'acc_id1')
rule_code = white_acc_id1[(white_acc_id1.UID_x>=1)&(white_acc_id1.UID_y>=6)].acc_id1.tolist()
test_rule_uid_12 = trans_test[trans_test['acc_id1'].isin(rule_code)].UID.unique()
# #acc_id2
# white_acc_id2= find_good(trans_train, 'acc_id2')
# rule_code = white_acc_id2[(white_acc_id2.UID_x>=0.99)&(white_acc_id2.UID_y>=200)].acc_id2.tolist()
# test_rule_uid_11 = trans_test[trans_test['acc_id2'].isin(rule_code)].UID.unique()
# #acc_id3
# white_acc_id3= find_good(trans_train, 'acc_id3')
# rule_code = white_acc_id3[(white_acc_id3.UID_x>=0.99)&(white_acc_id3.UID_y>=200)].acc_id3.tolist()
# test_rule_uid_11 = trans_test[trans_test['acc_id3'].isin(rule_code)].UID.unique()
#device_code1安卓设备号 覆盖8
white_device_code1= find_good(trans_train, 'device_code1')
rule_code = white_device_code1[(white_device_code1.UID_x>=1)&(white_device_code1.UID_y>=10)].device_code1.tolist()
test_rule_uid_13 = trans_test[trans_test['device_code1'].isin(rule_code)].UID.unique()
# #device_code3
# white_device_code3= find_good(trans_train, 'device_code3')
# rule_code = white_device_code3[(white_device_code3.UID_x>=1)&(white_device_code3.UID_y>=6)].device_code3.tolist()
# test_rule_uid_11 = trans_test[trans_test['device_code3'].isin(rule_code)].UID.unique()
#mac1操作设备MAC地址 覆盖15
white_mac1= find_good(trans_train, 'mac1')
rule_code = white_mac1[(white_mac1.UID_x>=1)&(white_mac1.UID_y>=20)].mac1.tolist()
test_rule_uid_14 = trans_test[trans_test['mac1'].isin(rule_code)].UID.unique()
#ip1 IP地址 覆盖55
white_ip1= find_good(trans_train, 'ip1')
rule_code = white_ip1[(white_ip1.UID_x>=1)&(white_ip1.UID_y>=20)].ip1.tolist()
test_rule_uid_15 = trans_test[trans_test['ip1'].isin(rule_code)].UID.unique()
#geo_code 经纬度编码 覆盖827个
white_geo_code= find_good(trans_train, 'geo_code')
rule_code = white_geo_code[(white_geo_code.UID_x>=1)&(white_geo_code.UID_y>=60)].geo_code.tolist()
test_rule_uid_16 = trans_test[trans_test['geo_code'].isin(rule_code)].UID.unique()
#market_code 营销活动号 覆盖295
white_market_code= find_good(trans_train, 'market_code')
rule_code = white_market_code[(white_market_code.UID_x>=1)&(white_market_code.UID_y>=100)].market_code.tolist()
test_rule_uid_17 = trans_test[trans_test['market_code'].isin(rule_code)].UID.unique()

#oper#################
# #device_code1 安卓设备码
# white_device_code1= find_good(op_train, 'device_code1')
# rule_code = white_device_code1[(white_device_code1.UID_x>=1)&(white_device_code1.UID_y>=20)].device_code1.tolist()
# test_rule_uid_23 = op_test[op_test['device_code1'].isin(rule_code)].UID.unique()
# #device_code3 苹果设备码
# white_device_code3= find_good(op_train, 'device_code3')
# rule_code = white_device_code3[(white_device_code3.UID_x>=1)&(white_device_code3.UID_y>=20)].device_code3.tolist()
# test_rule_uid_23 = op_test[op_test['device_code3'].isin(rule_code)].UID.unique()
#ip1 设备IP 覆盖317
white_ip1= find_good(op_train, 'ip1')
rule_code = white_ip1[(white_ip1.UID_x>=1)&(white_ip1.UID_y>=30)].ip1.tolist()
test_rule_uid_21 = op_test[op_test['ip1'].isin(rule_code)].UID.unique()
# #ip2 电脑IP
# white_ip2= find_good(op_train, 'ip2')
# rule_code = white_ip2[(white_ip2.UID_x>=1)&(white_ip2.UID_y>=10)].ip2.tolist()
# test_rule_uid_23 = op_test[op_test['ip2'].isin(rule_code)].UID.unique()
#mac1 MAC地址 覆盖11
white_mac1= find_good(op_train, 'mac1')
rule_code = white_mac1[(white_mac1.UID_x>=1)&(white_mac1.UID_y>=20)].mac1.tolist()
test_rule_uid_22 = op_test[op_test['mac1'].isin(rule_code)].UID.unique()
#mac2 WIFI地址 覆盖18
white_mac2= find_good(op_train, 'mac2')
rule_code = white_mac2[(white_mac2.UID_x>=1)&(white_mac2.UID_y>=20)].mac2.tolist()
test_rule_uid_23 = op_test[op_test['mac2'].isin(rule_code)].UID.unique()
#geo_code 经纬度编码 覆盖504
white_geo_code= find_good(op_train, 'geo_code')
rule_code = white_geo_code[(white_geo_code.UID_x>=1)&(white_geo_code.UID_y>=60)].geo_code.tolist()
test_rule_uid_24 = op_test[op_test['geo_code'].isin(rule_code)].UID.unique()

#规则应用
outpath = 'F:/数据集/1207甜橙金融/out/'
auc = 21
f = open('F:/数据集/1207甜橙金融/out/baseline_%s_badrule8.csv' % str(auc), encoding='utf8')
submit = pd.read_csv(f) #读取覆盖坏规则后的提交数据

test_rule_uid = pd.DataFrame(list(set(test_rule_uid_24)
                                  ))
pred_data_rule = submit.merge(test_rule_uid, left_on='UID', right_on=0, how='left')
pred_data_rule['Tag'][(pred_data_rule[0] > 0)] = 0
pred_data_rule[['UID', 'Tag']].to_csv(outpath+'baseline_{}_badrule8_goodrule3.csv'.format(auc), index=False)
