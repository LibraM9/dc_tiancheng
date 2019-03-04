# -*- coding: utf-8 -*-
#@author: limeng
#@file: tfidf.py
#@time: 2018/12/13 18:47
"""
文件说明：
"""
import pandas as pd
import numpy as np

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

x = op_train.groupby('UID')['mode'].apply(lambda x:' '.join(x.tolist()))

from sklearn.feature_extraction.text import CountVectorizer
train_set = x.tolist()
test_set = ("The sun in the sky is bright.","We can see the shining sun, the bright sun.")
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(train_set)
print( "Vocabulary:", count_vectorizer.vocabulary_)# Vocabulary: {'blue': 0, 'sun': 3, 'bright': 1, 'sky': 2}
# 得到词频矩阵
freq_term_matrix = count_vectorizer.transform(train_set)
print(freq_term_matrix.todense() )
# 注意：在最新的sklearn中idf是# smooth_idf = Talse, log(N / df) + 1，所以只要同时出现的话,就是1;
# smooth_idf = True, idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
print("IDF:", tfidf.idf_)
tf_idf_matrix = tfidf.transform(freq_term_matrix)
print(tf_idf_matrix.todense())

#countvectororizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=100, max_df=0.8)
vectorizer.fit_transform(op_train['mode'])
