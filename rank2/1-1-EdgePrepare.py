
# 选择维度作图

import pandas as pd
import numpy as np
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from multiprocessing import Pool
import time
import pickle
import os
from sklearn.preprocessing import LabelEncoder


dataroot = 'F:/数据集/1207甜橙金融/data/'
cacheRoot = 'F:/数据集处理/甜橙/'


trainFiles  = {"transaction":'transaction_train_new.csv', "operation" :'operation_train_new.csv', "label" :'tag_train_new.csv'} 
validFiles = {"transaction":'transaction_round1_new.csv', "operation" :'operation_round1_new.csv'}
testFiles  = {"transaction":'test_transaction_round2.csv', "operation" :'test_operation_round2.csv'}


# train operation
f = open(dataroot + trainFiles['operation'], encoding="utf8")
train_oper = pd.read_csv(f, encoding="utf8")
# train transaction
f = open(dataroot + trainFiles['transaction'], encoding="utf8")
train_transac = pd.read_csv(f,encoding="utf8")
# train tag
# test operation
f = open(dataroot + validFiles["operation"], encoding="utf8")
test_oper_r1 = pd.read_csv(f,encoding="utf8")
# test transaction
f = open(dataroot + validFiles["transaction"], encoding="utf8")
test_transac_r1 = pd.read_csv(f,encoding="utf8")

f = open(dataroot + testFiles["operation"], encoding="utf8")
test_oper_r2 = pd.read_csv(f,encoding="utf8")
# test transaction
f = open(dataroot + testFiles["transaction"], encoding="utf8")
test_transac_r2 = pd.read_csv(f,encoding="utf8")



oper_tran_grpuse_col = ['UID',"device1","mac1","ip1","geo_code", "device_code1","device_code2","device_code3" ]
oper_col = [ 'UID','ip','ip_sub','wifi' ]# ip  ip1 fillna ip2  ip1_sub fillna ip2_sub
tran_col = ['UID', 'merchant', 'code2' ]




# ['ip1','mac1','mac2','geo_code']
# ['ip1','mac1','mac2' ,'merchant']
oper_tran_grpuse = pd.concat([ train_oper[oper_tran_grpuse_col],test_oper_r1[oper_tran_grpuse_col] , test_oper_r2[oper_tran_grpuse_col],train_transac[oper_tran_grpuse_col],test_transac_r1[oper_tran_grpuse_col],test_transac_r2[oper_tran_grpuse_col]])
train_oper['ip']   = train_oper['ip1'].fillna( train_oper['ip2'])
test_oper_r1['ip'] = test_oper_r1['ip1'].fillna( test_oper_r1['ip2'])
test_oper_r2['ip'] = test_oper_r2['ip1'].fillna( test_oper_r2['ip2'])

train_oper['ip_sub']   = train_oper['ip1_sub'].fillna( train_oper['ip2_sub'])
test_oper_r1['ip_sub'] = test_oper_r1['ip1_sub'].fillna( test_oper_r1['ip2_sub'])
test_oper_r2['ip_sub'] = test_oper_r2['ip1_sub'].fillna( test_oper_r2['ip2_sub'])
oper_use = pd.concat([ train_oper,test_oper_r1,test_oper_r2 ])
tran_use = pd.concat([ train_transac,test_transac_r1,test_transac_r2 ])




sourcedata = [{"usedata":oper_tran_grpuse,"useCol":oper_tran_grpuse_col},
              {"usedata":oper_use,"useCol":oper_col},
              {"usedata":tran_use,"useCol":tran_col}
             ]

def create_edgelist(useData, secNodeCol):
    """
    :param useData:
    :param secNodeCol:
    :return: 变得权重
    """
    le = LabelEncoder()
    datacp = useData[["UID",secNodeCol ]]
    datacp = datacp[-datacp[secNodeCol].isnull()]
    datacp[secNodeCol] = le.fit_transform(datacp[secNodeCol]) + 1000000
    each =datacp.groupby(["UID", secNodeCol])['UID'].agg({"trans_cnt":'count'}).reset_index()
    total = each.groupby(secNodeCol)["trans_cnt"].agg({ 'trans_cnt_total':"sum"}).reset_index()
    gp = pd.merge(each, total, on=[secNodeCol])
    del datacp, each, total
    gp["ratio"] = gp['trans_cnt']/gp['trans_cnt_total']
    gp = gp.drop(['trans_cnt', 'trans_cnt_total'], axis=1)
    savename = cacheRoot + 'sourceEmb/{}_weighted_edglist_filytypeTxt.txt'.format(secNodeCol)
    np.savetxt(savename, gp.values, fmt=['%d','%d','%f'])
    gp = gp.drop("ratio",axis = 1)
    savenameForDeepWalk = cacheRoot + '{}_weighted_edglist_DeepWalk.txt'.format(secNodeCol)
    np.savetxt(savenameForDeepWalk, gp.values, fmt=['%d','%d'])
    del gp



for spec in sourcedata:
    for c in spec["useCol"]:
        if c != "UID":
            create_edgelist(spec["usedata"], c)
            gc.collect()


import networkx as nx
def createEdgeFomat(fname):
    """
    画图
    :param fname:
    :return:
    """
    G = nx.Graph()
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    lines =[l.replace("\n","").split(" ")  for l in lines]
    lines = [[int(x[0]),int(x[1]),float(x[2])] for x in lines]
    edfname = fname.replace(".txt",".edgelist")
    
    for edg in lines:
        G.add_edge(edg[0], edg[1], weight=edg[2])
    print("\n-------------------------------------\n")
    print("saving fali name %s " % edfname)
    print("\n-------------------------------------\n")
    fh=open(edfname,'wb')
    nx.write_edgelist(G, fh)
    fh.close()

for f in os.listdir(cacheRoot):
    if "DeepWalk" not in f:
        print("creating %s edge format for node2vec embedding ... " % (f.replace("_edglist_filytypeTxt.txt", "" )))
        createEdgeFomat(cacheRoot + f)
        print(f.split(".")[0],"finish")



