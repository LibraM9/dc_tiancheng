# -*- coding: utf-8 -*-
#@author: limeng
#@file: data_check.py
#@time: 2018/12/10 9:59
"""
文件说明：数据探查
"""
import pandas as pd

path = 'F:/数据集/1207甜橙金融/data/'
f = open(path+ 'tag_train_new.csv',encoding='utf8')
y = pd.read_csv(f)
f = open(path+ 'operation_train_new.csv',encoding='utf8')
operation = pd.read_csv(f)
f = open(path+ 'transaction_train_new.csv',encoding='utf8')
transaction = pd.read_csv(f)

#####bad
bad = y[y.Tag == 1]
oper_bad = operation[operation.UID.isin(bad.UID.values)]
oper_bad = oper_bad.sort_values(by=['UID'], ascending=(True))
trans_bad = transaction[transaction.UID.isin(bad.UID.values)]
trans_bad = trans_bad.sort_values(by=['UID'], ascending=(True))

x = oper_bad.sort_values(by=['geo_code','UID'], ascending=(True,True))
x = x[['UID','geo_code']]