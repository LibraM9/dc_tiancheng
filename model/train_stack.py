# -*- coding: utf-8 -*-
#@author: limeng
#@file: train_stack.py
#@time: 2018/12/14 8:58
"""
文件说明：模型集成训练
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def tpr_weight_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return 'TC_AUC', 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3, True

path = 'F:/数据集/1207甜橙金融/data/feature/'

f = open(path+'train1214.csv',encoding='utf8')
train = pd.read_csv(f)
f = open(path+'test1214.csv',encoding='utf8')
test = pd.read_csv(f)

label = train['Tag']
train = train.drop(['Tag','UID'], axis=1)

test_id = test['UID']
test = test.drop(['Tag','UID'], axis=1)

#缺失值填充
from model_train.a1_preprocessing import NumImputer
ni = NumImputer(strategy='median')
train_x = ni.fit_transform(train)
test_x = ni.transform(test)

#模型参数
from lightgbm import LGBMClassifier
lgb1 = LGBMClassifier(
    boosting='gbdt',
    application='binary',
    metric='auc',
    learning_rate=0.05,
    max_depth=17,#17
    num_leaves=4,#4
    min_child_samples=300,#280 300 500~1000
    min_split_gain=0.47,#0.176
    lambda_l1=0.5,#0.5
    lambda_l2=8.5,#8.5
    num_threads=-1,
    n_estimators=150,
    bagging_fraction=0.5, #0.5
    feature_fraction=0.3,#0.3
    is_unbalance=True,
)

from xgboost import XGBClassifier #xgboost分类
xgb = XGBClassifier(n_estimators=500,
                       learning_rate=0.02,
                       max_depth=4,
                       colsample_bytree=0.7,
                       colsample_bylevel=0.7,
                       subsample=0.7)
from sklearn.ensemble import RandomForestClassifier#随机森林分类
rf = RandomForestClassifier(n_estimators=300,
                            max_depth=6,
                            min_samples_split=100,
                            max_features='auto',
                            warm_start=True)
from sklearn.ensemble import ExtraTreesClassifier#极端树分类
etr = ExtraTreesClassifier(n_estimators=500,
                              max_depth=6,
                              max_features=0.5)

from sklearn.ensemble import GradientBoostingClassifier#GBDT分类
gbdt = GradientBoostingClassifier(learning_rate=0.02,
                                  n_estimators=500,
                                  max_depth=6)
#模型集成
from model_train.a3_model6 import ModelStacking
clfs = [lgb1,xgb,rf,etr,gbdt]
ms = ModelStacking(clfs)
data_stack_train = ms.fit(train_x, label)
data_stack_test_1 = ms.transform(test_x)

lgb2 = LGBMClassifier(
    boosting='gbdt',
    application='binary',
    metric='auc',
    learning_rate=0.05,
    max_depth=14,#17
    num_leaves=4,#4
    min_child_samples=300,#280 300 500~1000
    min_split_gain=0.1,#0.176
    lambda_l1=0.08,#0.5
    lambda_l2=8.5,#8.5
    num_threads=-1,
    n_estimators=150,
    bagging_fraction=0.5, #0.5
    feature_fraction=0.3,#0.3
    is_unbalance=True,
)
lgb2.fit(data_stack_train, label)
train_proba = lgb2.predict_proba(data_stack_train)[:, 1]
test_proba = lgb2.predict_proba(data_stack_test_1)[:, 1]
print(tpr_weight_funtion(train_proba))

from sklearn.model_selection import cross_val_score
cv = cross_val_score(lgb2, data_stack_train, label, cv=5, scoring='roc_auc')
print(cv)
print(np.mean(cv))

#训练，K折交叉拆分
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(data_stack_train.shape[0])
sub_preds = np.zeros(test_id.shape[0])
data_stack_train=pd.DataFrame(data_stack_train)
data_stack_test_1 = pd.DataFrame(data_stack_test_1)
for index, (train_index, test_index) in enumerate(skf.split(data_stack_train, label)):
    lgb2.fit(data_stack_train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(data_stack_train.iloc[train_index], label.iloc[train_index]),
                            (data_stack_train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
    best_score.append(lgb2.best_score_['valid_1']['auc'])
    print(best_score)
    oof_preds[test_index] = lgb2.predict_proba(data_stack_train.iloc[test_index], num_iteration=lgb2.best_iteration_)[:,
                            1]

    test_pred = lgb2.predict_proba(data_stack_test_1, num_iteration=lgb2.best_iteration_)[:, 1]
    sub_preds += test_pred / 5 #取5次预测的均值

    # print('test mean:', test_pred.mean())
    # predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

m = tpr_weight_funtion(y_predict=oof_preds, y_true=label)
print(m[1])

ans = pd.DataFrame([test_id, sub_preds]).T
ans.columns = ['UID', 'pred']
f = open('F:/数据集/1207甜橙金融/data/test_submit.csv',encoding='utf8')
sub = pd.read_csv(f)
sub = sub.merge(ans,on='UID', how='left')
sub['Tag'] = sub['pred']
sub = sub[['UID','Tag']]
outpath = 'F:/数据集/1207甜橙金融/out/'
sub.to_csv(outpath + 'baseline_%s.csv' % str(m[1])[2:6], index=False)