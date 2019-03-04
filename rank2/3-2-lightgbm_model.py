import datetime
import pandas as pd
import numpy as np
import gensim
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')


data = pd.read_csv('../cache/features_add.csv')
print(data.shape)


train = data[data['Tag']!=0.5].copy()
test = data[data['Tag']==0.5].copy()
y_train = train['Tag']
feats = [f for f in train.columns if f not in ['UID', 'Tag']]



def tpr_weight_funtion(y_true,y_predict):
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
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

def eval_function(y_predict,dtrain):
    y_true = dtrain.get_label()
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
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'tpr', 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3, True





# LightGBM


# 贝叶斯调参后的最优参数
parameters = {'boosting_type': 'gbdt',
              'objective': 'binary',
              'learning_rate': 0.04,
              'metric': 'binary_logloss',
              'num_leaves': 112,
              'max_depth': 6,
              'feature_fraction': 0.7,
              'bagging_fraction': 0.77,
              'subsample_freq': 1,
              'seed': 666,
              'verbose': -1,
              'n_jobs': 10,
              'lambda_l2': 0.84,
              'lambda_l1': 7.68,
#               'max_bin': 310
              }
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
xx_submit = []
xx_tpr = []
xx_auc = []
xx_iteration = []
oof_preds = np.zeros(train.shape[0])
feature_importance_df = pd.DataFrame()

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train['Tag'])):
    dtrain = lgb.Dataset(data=train[feats].iloc[train_idx],
                         label=train['Tag'].iloc[train_idx])
    dvalid = lgb.Dataset(data=train[feats].iloc[valid_idx],
                         label=train['Tag'].iloc[valid_idx])
    clf = lgb.train(
        params=parameters,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        early_stopping_rounds=100,
        verbose_eval=False,
#         feval=eval_function
    )
    # save feature's importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)
    valid_preds = clf.predict(train[feats].iloc[valid_idx], num_iteration=clf.best_iteration)
    print('Fold%2d LOGLOSS: %.6f' % (n_fold + 1, clf.best_score['valid_0']['binary_logloss']), 'Fold%2d TPR: %.6f' % (n_fold + 1, tpr_weight_funtion(train['Tag'][valid_idx], valid_preds)))
    xx_auc.append(clf.best_score['valid_0']['binary_logloss'])
    xx_tpr.append(tpr_weight_funtion(train['Tag'][valid_idx], valid_preds))
    xx_iteration.append(clf.best_iteration)
    xx_submit.append(clf.predict(test[feats], num_iteration=clf.best_iteration))
    oof_preds[valid_idx] = clf.predict(train[feats].iloc[valid_idx], num_iteration=clf.best_iteration)

print('特征个数:%d' % (len(feats)))
print('线下平均LOGLOSS:%.5f' % (np.mean(xx_auc)))
print('线下全集TPR:%.5f' % (tpr_weight_funtion(train['Tag'], oof_preds)))
print('线下平均TPR:%.5f' % (np.mean(xx_tpr)))
print('线下平均迭代次数:%d' % (np.mean(xx_iteration)))
print(xx_iteration)



# 特征重要性
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False)[:50].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize = (15, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../cache/lgbm_importances.png')
display_importances(feature_importance_df)
feature_importance_df.to_csv('../cache/feature_importance_df.csv', index=False)



# 提交结果

s = 0
for i in xx_submit:
    s = s + i

test['Tag'] = list(s / 5)
test = data[data['Tag']==0.5].copy()
submission = test[['UID', 'Tag']]
submission[['UID', 'Tag']].to_csv("../submission/lgb_basesub.csv", index=False)

