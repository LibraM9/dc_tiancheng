# -*- coding: utf-8 -*-
#@author: limeng
#@file: train.py
#@time: 2018/12/10 14:23
"""
文件说明：模型训练
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
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

f = open(path+'train1210.csv',encoding='utf8')
train = pd.read_csv(f)
f = open(path+'test1210.csv',encoding='utf8')
test = pd.read_csv(f)

label = train['Tag']
train = train.drop(['Tag','UID'], axis=1)

from model_train.a2_feature_selection import iv
iv(train,label, NUM_BIN = 15,OUTPUT_PATH='F:/数据集/1207甜橙金融/feature/')

test_id = test['UID']
test = test.drop(['Tag','UID'], axis=1)

lgb_model = lgb.LGBMClassifier(
     boosting_type='gbdt'
    , num_leaves=100
    , reg_alpha=3
    , reg_lambda=5
    , max_depth=-1
    , n_estimators=500
    , objective='binary'
    , subsample=0.9
    , colsample_bytree=0.77
    , subsample_freq=1
    , learning_rate=0.05
    ,random_state=1000
    , n_jobs=16
    , min_child_weight=4
    , min_child_samples=5
    , min_split_gain=0)

#交叉验证调参
# from sklearn.model_selection import train_test_split
# x, val_x, y, val_y = train_test_split(
#     train,
#     label,
#     test_size=0.3,
#     random_state=1,
#     stratify=train  # 这里保证分割后y的比例分布与原数据一致
# )
from sklearn.model_selection import cross_val_score
cv = cross_val_score(lgb_model, train, label, cv=5, scoring='roc_auc')
print(cv)
print(np.mean(cv))

#训练，K折交叉拆分
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])

for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,
                            1]

    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
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


