import pandas as pd
import numpy as np
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
from sklearn.svm import SVC
warnings.simplefilter(action='ignore', category=FutureWarning)
dataroot = "../data/"
cacheRoot = "../cache/"
subRoot = "../submission/"


# import os

# def list_files(startpath):
#     for root, dirs, files in os.walk(startpath):
#         level = root.replace(startpath, '').count(os.sep)
#         indent = ' ' * 4 * (level)
#         print('{}{}/'.format(indent, os.path.basename(root)))
#         subindent = ' ' * 4 * (level + 1)
#         for f in files:
#             print('{}{}'.format(subindent, f))
#list_files("../")
def get_embeding(fname, embname):
    f = open(fname)
    embeding_lines = f.readlines()
    f.close()
    mapfunc = lambda x: list( map( float, x ) )
    embeding_lines = [li.replace("\n","").split(" ") for li in embeding_lines[1:]]
    embeding_lines = [ [ int(line[0]) ] +  mapfunc( line[1:]  )   for line in   embeding_lines ]
    cols = ["UID"] + [ embname  + str(i) for i in range( len(embeding_lines[0]) -1 )]
    embeding_df = pd.DataFrame(embeding_lines, columns=cols )
    del embeding_lines
    return embeding_df



data =  pd.read_csv(cacheRoot + "grouping_features.csv")


mac1_emb = get_embeding(cacheRoot + "mac1_.emb", "mac1_emb_")
merchant_dbk = get_embeding(cacheRoot + "merchant_weighted_edglist_DeepWalk.embeddings", "merchant_deepwalk_")
mac1_emb = mac1_emb[ mac1_emb.UID.map(lambda x: x<= 131587) ] 
merchant_dbk = merchant_dbk[ merchant_dbk.UID.map(lambda x: x<= 131587) ] 


merchant_emb = get_embeding(cacheRoot + "merchant_.emb", "merchant_emb_")
merchant_emb = merchant_emb[ merchant_emb.UID.map(lambda x: x<= 131587) ] 
mac_merch = pd.DataFrame()
mac_merch["UID"] = merchant_dbk.UID.tolist() +  mac1_emb.UID.tolist()
mac_merch = mac_merch.merge( mac1_emb, on = ["UID"],how = "left"  )
mac_merch = mac_merch.merge( merchant_dbk, on = ["UID"],how = "left"  )
mac_merch = mac_merch.fillna( 0.0 )


# for embedding mac1 and merchants pca
from sklearn.decomposition import PCA
dim = 42
pca = PCA(n_components=dim)
pca_res = pca.fit_transform(mac_merch.drop("UID", axis = 1)  )
me_pca = pd.DataFrame(pca_res, columns=["pca_mac1_merchrant_%d" % i for i in range(dim)])
me_pca["UID"] = mac_merch.UID.values



datapca = pd.merge( data,me_pca,on = ["UID"],how = "left"  )
data = pd.merge( data, mac1_emb, on =["UID"], how="left" )
data = pd.merge( data, merchant_dbk, on =["UID"], how="left" )


merchant_dbk.head()


import os
import pandas as pd
import numpy as np
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.svm import SVC
def  process_feature(train_x, valid_x, test_df):
    result = []
    drop_cols = ['Tag']
    for df in [train_x, valid_x, test_df]:
        result.append(df.drop(drop_cols, axis=1))
    return result 
def cv(df, num_folds, param, stratified=True, debug=False):
    train_df =  df[df.Tag != 0.5]
    test_df =   df[df.Tag == 0.5]
    
    seed = 178
    if "seed" in param: 
        seed = param["seed"]
        del param['seed']
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(train_df.shape[0])
    all_test_preds = []    
    feature_importance_df = pd.DataFrame()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['Tag'])):
        train_x, train_y = train_df.iloc[train_idx], train_df['Tag'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_df['Tag'].iloc[valid_idx]
        fold_preds = test_df[["UID"]]
        
        train_x, valid_x, test = process_feature(train_x, valid_x, test_df)
        if n_fold == 0:
            print(train_x.shape, valid_x.shape, test.shape)
        
        train_data = lgb.Dataset(train_x, label=train_y)
        validation_data = lgb.Dataset(valid_x, label=valid_y)

        clf=lgb.train(param,
                      train_data,
                      num_boost_round=10000,
                      valid_sets=[train_data, validation_data],
                      valid_names=["train", "valid"],
                      early_stopping_rounds=100,
                      verbose_eval=100)

        valid_preds = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_preds = clf.predict(test, num_iteration=clf.best_iteration)

        fold_preds['Tag'] = test_preds
        fold_preds['fold_id'] = n_fold + 1
        all_test_preds.append(fold_preds)
        
        oof_preds[valid_idx] = valid_preds
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, valid_preds)))
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    full_auc = roc_auc_score(train_df['Tag'], oof_preds)
    all_test_preds = pd.concat(all_test_preds, axis=0)
    sub = pd.DataFrame()
    sub['UID'] = all_test_preds.UID.unique()
    sub.set_index("UID", inplace=True)
    sub["Tag"] = all_test_preds.groupby("UID").Tag.mean()
    print('Full AUC score %.6f' % full_auc)
    
    return [full_auc,sub]



class runmodel(object):
    def __init__(self,data):
        self.data = data
    def getinfo(self):
        print("use variable = runlgb.get_result() get reusult ")
    def get_result(self):
        params_list =[
         {'boosting_type': 'goss', 'colsample_bytree': 0.6555004575000242, 'learning_rate': 0.016380004820033073, 'max_bin': 1000, 'metric': 'auc', 'min_child_weight': 2.1115838168176433, 'num_leaves': 108, 'reg_alpha': 23.247001339889128, 'reg_lambda': 997.9576062039534, 'subsample': 1.0, 'verbose': 1, 'seed': 4419},
         {'boosting_type': 'goss', 'colsample_bytree': 0.7187703092392053, 'learning_rate': 0.01939219215282862, 'max_bin': 310, 'metric': 'auc', 'min_child_weight': 2.66983907940641, 'num_leaves': 89, 'reg_alpha': 18.48224434106526, 'reg_lambda': 470.54675380054465, 'subsample': 1.0, 'verbose': 1, 'seed': 57},
         {'boosting_type': 'goss', 'colsample_bytree': 0.6274617979582582, 'learning_rate': 0.01680918441243103, 'max_bin': 780, 'metric': 'auc', 'min_child_weight': 0.8226071606806127, 'num_leaves': 73, 'reg_alpha': 14.466924422050258, 'reg_lambda': 658.5772060624658, 'subsample': 1.0, 'verbose': 1, 'seed': 1732},
         {'boosting_type': 'goss', 'colsample_bytree': 0.6991838451153098, 'learning_rate': 0.01577419276366034, 'max_bin': 940, 'metric': 'auc', 'min_child_weight': 7.758954855388241, 'num_leaves': 187, 'reg_alpha': 43.60868666926589, 'reg_lambda': 667.6371302027073, 'subsample': 1.0, 'verbose': 1, 'seed': 139},
         {'boosting_type': 'goss', 'colsample_bytree': 0.6198406436401038, 'learning_rate': 0.016662891953242748, 'max_bin': 820, 'metric': 'auc', 'min_child_weight': 3.5602833459924015, 'num_leaves': 77, 'reg_alpha': 13.398041512170746, 'reg_lambda': 631.8105595021391, 'subsample': 1.0, 'verbose': 1, 'seed': 7609}
        ]
        result = []
        for i, params in enumerate(params_list):
            model_dir = "./model_output/random/lgb_693/%d/" % i
            result.append(cv(self.data , 5, params))
        return result



runlgb = runmodel(data)
runlgb.getinfo()
sub_embeding = runlgb.get_result()

all_subs = [x[1] for x in sub_embeding]
all_sub3 = pd.concat( all_subs  )
all_sub3 = all_sub3.groupby("UID")["Tag"].agg({"Tag":"mean" }).reset_index()
all_sub3.Tag = all_sub3.Tag * 2.5
all_sub3.to_csv(subRoot + "sub2.csv", encoding = 'utf8', index = False)




runlgbpca = runmodel(datapca)
runlgbpca.getinfo()
sub_embedingpca = runlgbpca.get_result()

all_subspca = [x[1] for x in sub_embedingpca]
all_subspca = pd.concat( all_subs  )
all_subspca = all_subspca.groupby("UID")["Tag"].agg({"Tag":"mean" }).reset_index()
all_subspca.to_csv(subRoot + "sub3.csv", encoding = 'utf8', index = False)



