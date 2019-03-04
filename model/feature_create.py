# -*- coding: utf-8 -*-
#@author: limeng
#@file: train.py
#@time: 2018/12/10 9:00
"""
文件说明：特征构造
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

path = 'F:/数据集/1207甜橙金融/data/'

f = open(path+ '开放数据_甜橙金融杯数据建模/operation_TRAIN.csv',encoding='utf8')
op_train = pd.read_csv(f)
f = open(path+ '开放数据_甜橙金融杯数据建模/transaction_TRAIN.csv',encoding='utf8')
trans_train = pd.read_csv(f)

f = open(path+ 'test_operation_round2.csv',encoding='utf8')
op_test = pd.read_csv(f)
f = open(path+ 'test_transaction_round2.csv',encoding='utf8')
trans_test = pd.read_csv(f)

f = open(path+ '开放数据_甜橙金融杯数据建模/tag_TRAIN.csv',encoding='utf8')
y = pd.read_csv(f)
f = open(path+ 'test_submit.csv',encoding='utf8')
sub = pd.read_csv(f)

def calc_ent(x):
    """ calculate shanno ent of x """
    x_value_list = x.unique()
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
        return ent

def get_feature(op, trans, label):
    for feature in op.columns[:]:
        if feature not in ['day']:
            if feature != 'UID':
                label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
                label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
                label = label.merge(op.groupby(['UID'])[feature].apply(calc_ent).reset_index(), on='UID', how='left')
            for deliver in ['ip1', 'mac1', 'mac2', 'geo_code']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                    else:
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                   how='left')[['UID_x', 'UID_y']]
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = \
                        op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                   how='left')[['UID_x', 'UID_y']]
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp

        else:
            print(feature)
            label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
            for deliver in ['ip1', 'mac1', 'mac2']:
                if feature not in deliver:
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    op[['UID', deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                               how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].max().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].min().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].sum().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].mean().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = op[['UID', deliver]].merge(op.groupby([deliver])[feature].std().reset_index(), on=deliver,
                                                      how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp

    for feature in trans.columns[1:]:
        if feature not in ['trans_amt', 'bal', 'day']:
            if feature != 'UID':
                label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
                label = label.merge(trans.groupby(['UID'])[feature].apply(calc_ent).reset_index(), on='UID', how='left')
            for deliver in ['merchant', 'ip1', 'mac1', 'geo_code', ]:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                             on=deliver, how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                             on=deliver, how='left')[['UID', feature]]
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                    else:
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),
                                                             on=deliver, how='left')[['UID_x', 'UID_y']]
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
                        temp = trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),
                                                             on=deliver, how='left')[['UID_x', 'UID_y']]
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID', feature + deliver]
                        label = label.merge(temp, on='UID', how='left')
                        del temp
            # if feature in ['merchant','code2','acc_id1','market_code','market_code']:
            #    label[feature+'_z'] = 0
            #    label[feature+'_z'] = label[feature+'_y']/label[feature+'_x']
        else:
            print(feature)
            label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
            for deliver in ['merchant', 'ip1', 'mac1']:
                if feature not in deliver:
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].max().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].min().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].sum().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].mean().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp
                    temp = \
                    trans[['UID', deliver]].merge(trans.groupby([deliver])[feature].std().reset_index(), on=deliver,
                                                  how='left')[['UID', feature]]
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID', feature + deliver]
                    label = label.merge(temp, on='UID', how='left')
                    del temp

    print("Done")
    return label

# train = get_feature(op_train, trans_train, y).fillna(-1)
# test = get_feature(op_test, trans_test, sub).fillna(-1)
train = get_feature(op_train, trans_train, y)
test = get_feature(op_test, trans_test, sub)

#时间
def other_feature(op_train,train):
    op_train['time_new'] = op_train['time'].apply(lambda x: int(x[:2]))
    feature = 'time_new'
    train = train.merge(op_train.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
    train = train.merge(op_train.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
    train = train.merge(op_train.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
    train = train.merge(op_train.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left')
    train = train.merge(op_train.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
    train = train.merge(op_train.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left')
    train = train.merge(op_train.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
    return train

train_plus = other_feature(op_train,train)
train_plus = other_feature(trans_train,train_plus)
test_plus = other_feature(op_test,test)
test_plus = other_feature(trans_test,test_plus)

train_plus.to_csv('F:/数据集/1207甜橙金融/data/feature/train1214.csv',encoding='UTF8',index=None)
test_plus.to_csv('F:/数据集/1207甜橙金融/data/feature/test1214.csv',encoding='UTF8',index = None)
