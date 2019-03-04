import pandas as pd
import numpy as np
import gensim
import warnings
warnings.filterwarnings('ignore')
np.random.seed(2018)

operation_df = pd.read_csv('../cache/operation_pre.csv', encoding='gbk')
transaction_df = pd.read_csv('../cache/transaction_pre.csv', encoding='gbk')
tag_trn = pd.read_csv('../data/tag_train_new.csv')
submission1 = pd.read_csv('../data/提交样例.csv')
submission2 = pd.read_csv('../data/submit_example.csv')

data = pd.concat([tag_trn, submission2], axis=0, ignore_index=True)


# 工具函数
# 计算两个序列之间重合度
def calculate_intersection_ratio(x, y):
    try:
        inter = x.intersection(y)
        union = x.union(y)
        return 1.0*len(inter)/len(union)
    except:
        return np.nan
# 计算行为A到行为B时间间隔小于阈值的个数
def getActionTimeSpan(df_action_of_userid, actiontypeA, actiontypeB, timethred):
    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid) - 1):
        if df_action_of_userid['action_type'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTimestamp'].iat[i]
            for j in range(i + 1, len(df_action_of_userid)):
                if df_action_of_userid['action_type'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTimestamp'].iat[j]
                if df_action_of_userid['action_type'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTimestamp'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
                    break
        i += 1
    return np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0)
# 计算用户当前行为与前一行为的所使用的设备及所处环境的吻合程度
def check_has_changed(df, df_feats, key, value, n, name):
    df_feats = df_feats.sort_values(by=['UID', 'day', 'time'])
    data_temp = df_feats[key+['day', 'time', value]].copy()
    shift_value = data_temp.groupby(key)[['day', 'time', value]].shift(n)
    shift_value = shift_value.rename(columns={'day': 'bef_day',
                                     'time': 'bef_time',
                                     value: 'bef_'+value})
    data_temp = data_temp.rename(columns={'day': 'cur_day',
                                  'time': 'cur_time',
                                  value: 'cur_'+value})
    data_temp = pd.concat([data_temp, shift_value], axis=1)
    data_temp[name] = 0
    data_temp.loc[data_temp['cur_'+value]!=data_temp['bef_'+value], name] = 1
    data_temp.loc[(data_temp['cur_'+value].isnull()==True)&(data_temp['bef_'+value].isnull()==True), name] = 0
    data_temp['day_diff'] = data_temp['cur_day']-data_temp['bef_day']
    data_temp['time_diff'] = data_temp.apply(lambda x: time_diff(x['cur_time'], x['bef_time']), axis=1)
    data_temp['time_diff'] = 1440.0*data_temp['day_diff']+data_temp['time_diff']
    # 全局
    gp = data_temp.groupby(key)[name].agg({name+'_count': 'sum',
                                           name+'_rt': 'mean'}).reset_index()
    df = df.merge(gp, on=['UID'], how='left')
    # 在1小时内
    tmp = data_temp[data_temp['time_diff']<=60].copy()
    gp = tmp.groupby(key)[name].agg({name+'_count_in_one_hour': 'sum',
                                     name+'_rt_in_one_hour': 'mean'}).reset_index()
    df = df.merge(gp, on=['UID'], how='left')
    # 在1分钟内
    tmp = data_temp[data_temp['time_diff']<=1].copy()
    gp = tmp.groupby(key)[name].agg({name+'_count_in_one_minute': 'sum',
                                     name+'_rt_in_one_minute': 'mean'}).reset_index()
    df = df.merge(gp, on=['UID'], how='left')
    return df
# word2vec特征
def w2v_features(df, df_feats, key, value, word_ndim,colname):
    df_feats = df_feats.sort_values(by=['UID', 'actionTime'])
    val_list = df_feats.groupby(key).apply(lambda x: x[value].tolist()).reset_index()
    val_list.columns = ['UID', value+'_list']
    val_list[value+'_list'] = val_list[value+'_list'].apply(lambda x: [str(f) for f in x])

    model = gensim.models.Word2Vec(val_list[value+'_list'], size=word_ndim, window=10, min_count=2, workers=4)
    model.save('../cache/'+value+'_word2vec.model')
    
    vocab = list(model.wv.vocab.keys())
    w2c_arr = []
    for v in vocab :
        w2c_arr.append(list(model.wv[v]))
        
    df_w2c_start = pd.DataFrame()
    df_w2c_start[value] = vocab
    df_w2c_start = pd.concat([df_w2c_start, pd.DataFrame(w2c_arr)], axis=1)
    df_w2c_start.columns = [value] + [colname+str(i) for i in range(word_ndim)]

    df_feats[value] = df_feats[value].astype(str)
    tmp = df_feats.merge(df_w2c_start, on=[value], how='left')
    gp = tmp.groupby(['UID'])[[colname+str(i) for i in range(word_ndim)]].agg({'mean', 'skew'})
    gp.columns = pd.Index([e[0] + "_" + e[1] for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    df = df.merge(gp, on=['UID'], how='left')
    return df
# 计算分组时间差
def group_diff_time(data, key, value, n, name):
    data_temp = data[key + [value]].copy()
    shift_value = data_temp.groupby(key)[value].shift(n)
    data_temp[name] = data_temp[value] - shift_value
    return data_temp[name]

"""
		特征工程

		operation+transaction

		用户操作（成功，失败，缺失）频次/交易频次/行为频次，频率，转化率
		用户操作/交易/整体使用设备种类个数/差值/重合度
		用户操作/交易/整体使用环境种类个数/差值/重合度
		行为设备/环境缺失个数平均值、最大值
		用户行为地点发生在中国境内的频次及频率
		用户当前行为是否为常用设备(device1,device2,device_code,geo_code,province,city,district,ip,ip_sub,mac1),统计频次及频率
		用户第一次操作/交易/行为时间，最后一次操作/交易/行为时间(actionTimestamp)
		用户每天操作/交易/行为频次、转化率统计(mean,max,min,std,max-min,skew)
		用户操作到交易时间间隔小于100秒的次数
		用户每天交易第一个时间到最后一个时间时间差(mean, max, min, std)
		设备、环境的热度(mean,max,min,skew,std,sum)，设备被用户操作的时间间隔(mean,max,min,std,skew,sum)
		用户行为/操作/交易天数
		用户当前行为发生设备、环境与前1行为的差异频次
		设备、ip第一次出现时间、最后一次出现时间的时间差
		用户同一天使用的操作类型、操作系统、版本、wifi的种类个数
		用户交易时间和操作时间的重合度
		用户行为时间的统计特征
		设备的w2v特征
		以用户最后一次行为时间做滑窗统计操作/交易/转化率特征
		用户使用不同设备时间长度
		用户行为时间差


"""


operation_df['action_type'] = 1
transaction_df['action_type'] = 2
transaction_df['success'] = 1
inner_cols = list(set(operation_df).intersection(set(transaction_df))) # 至合并operation_df和transaction_df有共同的特征列
user_action = pd.concat([operation_df[inner_cols], transaction_df[inner_cols]], axis=0, ignore_index=True)
user_action = user_action.sort_values(by=['UID', 'actionTime'])
operation_df = operation_df.sort_values(by=['UID', 'actionTime'])
transaction_df = transaction_df.sort_values(by=['UID', 'actionTime'])


gp = user_action.groupby(['UID', 'action_type']).size().unstack().reset_index().fillna(0)
gp.columns = ['UID', 'uid_operation_cnt', 'uid_trade_cnt']
gp['uid_action_cnt'] = gp['uid_operation_cnt'] + gp['uid_trade_cnt']
gp['uid_trade_ratio'] = gp['uid_trade_cnt'] / gp['uid_action_cnt']
gp['uid_trade_operation_ratio'] = gp['uid_trade_ratio'] / (0.01+gp['uid_operation_cnt'])
data = data.merge(gp, on=['UID'], how='left').fillna(0)

stats = user_action[user_action['action_type']==1].copy()
gp = stats.groupby(['UID', 'success']).size().unstack().reset_index().fillna(0)
gp.columns = ['UID','uid_unknown_operation_cnt', 'uid_failure_operation_cnt', 'uid_success_operation_cnt']
gp['uid_operation_cnt_diff'] = gp['uid_success_operation_cnt']-gp['uid_failure_operation_cnt']
data = data.merge(gp, on=['UID'], how='left')
data['uid_success_operation_ratio'] = data['uid_success_operation_cnt'] / (0.01+data['uid_operation_cnt'])
data['uid_failure_operation_ratio'] = data['uid_failure_operation_cnt'] / (0.01+data['uid_operation_cnt'])
data['uid_unknown_operation_ratio'] = data['uid_unknown_operation_cnt'] / (0.01+data['uid_operation_cnt'])
data['uid_trade_success_operation_ratio'] = data['uid_trade_ratio'] / (0.01+data['uid_success_operation_cnt'])
data['uid_trade_failure_operation_ratio'] = data['uid_trade_ratio'] / (0.01+data['uid_failure_operation_cnt'])
data['uid_trade_unknown_operation_ratio'] = data['uid_trade_ratio'] / (0.01+data['uid_unknown_operation_cnt'])
del data['uid_unknown_operation_cnt'], data['uid_trade_unknown_operation_ratio'], data['uid_unknown_operation_ratio']# 新的测试集里success没有缺失的情况




device_cols = ['device1', 'device2', 'device_code', 'mac1', 'device_brand']
tmp1 = user_action[user_action['action_type']==1].copy()
tmp2 = user_action[user_action['action_type']==2].copy()
for col in device_cols:
    gp = user_action.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_action_'+col+'_nunique'})
    data = data.merge(gp, on=['UID'], how='left')
    gp1 = tmp1.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_operation_'+col+'_nunique'})
    data = data.merge(gp1, on=['UID'], how='left')
    gp2 = tmp2.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_trade_'+col+'_nunique'})
    data = data.merge(gp2, on=['UID'], how='left')
    data['uid_trade_operation_'+col+'_diff'] = data['uid_trade_'+col+'_nunique'] - data['uid_operation_'+col+'_nunique']
for col in device_cols:
    gp1 = tmp1.groupby(['UID']).apply(lambda x: set(x[col].tolist())).reset_index()
    gp2 = tmp2.groupby(['UID']).apply(lambda x: set(x[col].tolist())).reset_index()
    gp1.columns = ['UID', 'uid_operation_'+col+'_list']
    gp2.columns = ['UID', 'uid_trade_'+col+'_list']
    data = data.merge(gp1, on=['UID'], how='left')
    data = data.merge(gp2, on=['UID'], how='left')
    data['uid_trade_operation_'+col+'_intersection_ratio'] = data.apply(lambda x: calculate_intersection_ratio(x['uid_operation_'+col+'_list'], x['uid_trade_'+col+'_list']), axis=1)    
    del data['uid_operation_'+col+'_list'], data['uid_trade_'+col+'_list']



    env_cols = ['ip', 'ip_sub', 'geo_code', 'nation', 'city', 'district']
tmp1 = user_action[user_action['action_type']==1].copy()
tmp2 = user_action[user_action['action_type']==2].copy()
for col in env_cols:
    gp = user_action.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_action_'+col+'_nunique'})
    data = data.merge(gp, on=['UID'], how='left')
    gp1 = tmp1.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_operation_'+col+'_nunique'})
    data = data.merge(gp1, on=['UID'], how='left')
    gp2 = tmp2.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_trade_'+col+'_nunique'})
    data = data.merge(gp2, on=['UID'], how='left')
    data['uid_trade_operation_'+col+'_diff'] = data['uid_trade_'+col+'_nunique'] - data['uid_operation_'+col+'_nunique']
for col in env_cols:
    gp1 = tmp1.groupby(['UID']).apply(lambda x: set(x[col].tolist())).reset_index()
    gp2 = tmp2.groupby(['UID']).apply(lambda x: set(x[col].tolist())).reset_index()
    gp1.columns = ['UID', 'uid_operation_'+col+'_list']
    gp2.columns = ['UID', 'uid_trade_'+col+'_list']
    data = data.merge(gp1, on=['UID'], how='left')
    data = data.merge(gp2, on=['UID'], how='left')
    data['uid_trade_operation_'+col+'_intersection_ratio'] = data.apply(lambda x: calculate_intersection_ratio(x['uid_operation_'+col+'_list'], x['uid_trade_'+col+'_list']), axis=1)    
    del data['uid_operation_'+col+'_list'], data['uid_trade_'+col+'_list']


    gp = user_action.groupby(['UID'])[['device_miss_cnt', 'env_miss_cnt']].agg({'mean', 'max'})
gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')

used_cols = ['ip_sub', 'device1', 'mac1','ip',
             'device_code', 'device2', 'geo_code']
for col in used_cols:
    gp = user_action.groupby(['UID'])[col].count().reset_index().rename(columns={col: 'uid_action_nonan_'+col+'_count'})
    data = data.merge(gp, on=['UID'], how='left')
    data['uid_action_nonan_'+col+'_ratio'] = data['uid_action_nonan_'+col+'_count']/(0.01+data['uid_action_cnt'])




tmp = user_action[user_action['is_china']>=0].copy()
gp = tmp.groupby(['UID'])['is_china'].agg({'uid_action_in_china_count':'sum',
                                           'uid_action_in_china_ratio':'mean'}).reset_index()
data = data.merge(gp, on=['UID'], how='left')


used_cols = ['device1', 'ip', 'ip_sub', 'mac1', 'device2','device_code','geo_code','province','city','district']
stats = user_action.copy()
for col in used_cols:
    gp = user_action.groupby(['UID', col]).size().reset_index().rename(columns={0: 'uid_action_favor_'+col+'_count'})
    gp = gp.sort_values(by=['UID', 'uid_action_favor_'+col+'_count'])
    gp = gp.groupby(['UID']).last().reset_index().rename(columns={col: 'uid_action_favor_'+col})
    stats = stats.merge(gp, on=['UID'], how='left')
    stats['is_action_favor_'+col] = 0
    stats.loc[stats['uid_action_favor_'+col]==stats[col], 'is_action_favor_'+col] = 1
    gp = stats.groupby(['UID'])['is_action_favor_'+col].agg({'is_action_favor_'+col+'_count': 'sum',
                                                             'is_action_favor_'+col+'_mean': 'mean'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')



gp = user_action.groupby(['UID'])['actionTimestamp'].agg({'uid_action_last_actionTimestamp': 'max',
                                                          'uid_action_first_actionTimestamp': 'min'}).reset_index()
gp['uid_actionTimestamp_timedelta'] = gp['uid_action_last_actionTimestamp'] - gp['uid_action_first_actionTimestamp']
data = data.merge(gp, on=['UID'], how='left')

gp = user_action[user_action['action_type']==2].groupby(['UID'])['actionTimestamp'].agg({'uid_trade_last_actionTimestamp': 'max',
                                                                                         'uid_trade_first_actionTimestamp': 'min'}).reset_index()
gp['uid_trade_actionTimestamp_timedelta'] = gp['uid_trade_last_actionTimestamp'] - gp['uid_trade_first_actionTimestamp']
data = data.merge(gp, on=['UID'], how='left')


stats = user_action.groupby(['UID','day','action_type',]).size().unstack().fillna(0).reset_index()
stats.columns = ['UID', 'day', 'uid_operation_day_count', 'uid_trade_day_count']
stats['uid_action_day_count'] = stats['uid_operation_day_count']+stats['uid_trade_day_count']
stats['uid_trade_day_ratio'] = stats['uid_trade_day_count']/stats['uid_action_day_count']
gp = stats.groupby(['UID'])[['uid_action_day_count', 'uid_operation_day_count', 'uid_trade_day_count', 'uid_trade_day_ratio']].agg({'max',
                                                                                                                                   'min',
                                                                                                                                   'std',
                                                                                                                                   'mean',
                                                                                                                                   'skew'})
gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')
data['uid_action_day_count_diff'] = data['uid_action_day_count_max'] - data['uid_action_day_count_min']
data['uid_trade_day_count_diff'] = data['uid_trade_day_count_max'] - data['uid_trade_day_count_min']
data['uid_operation_day_count_diff'] = data['uid_operation_day_count_max'] - data['uid_operation_day_count_min']
data['uid_trade_day_ratio_diff'] = data['uid_trade_day_ratio_max'] - data['uid_trade_day_ratio_min']



user_action = user_action.sort_values(by=['actionTime'])
userid = user_action['UID'].unique()
timespancount_dict = {'UID': [],
                      'uid_operation_to_trade_timdelta_count': []}
for uid in userid:
    action_df = user_action[user_action['UID']==uid].copy()
    actiontimespancount = getActionTimeSpan(action_df, 1, 2, timethred = 100)
    timespancount_dict['UID'].append(uid)
    timespancount_dict['uid_operation_to_trade_timdelta_count'].append(actiontimespancount)
timespancount_dict = pd.DataFrame(timespancount_dict)
data = data.merge(timespancount_dict, on=['UID'], how='left')



stats = user_action[user_action['action_type']==2].copy()
gp = stats.groupby(['UID', 'day'])['actionTimestamp'].agg({'uid_trade_day_last_actionTimestamp': 'max',
                                                           'uid_trade_day_first_actionTimestamp':'min'}).reset_index()
gp['uid_trade_day_timedelta'] = gp['uid_trade_day_last_actionTimestamp']-gp['uid_trade_day_first_actionTimestamp']
gp = gp.groupby(['UID'])['uid_trade_day_timedelta'].agg({'uid_trade_day_timedelta_mean': 'mean',
                                                         'uid_trade_day_timedelta_max': 'max',
                                                         'uid_trade_day_timedelta_min': 'min',
                                                         'uid_trade_day_timedelta_std': 'std',
                                                         'uid_trade_day_timedelta_skew': 'skew'}).reset_index()
data = data.merge(gp, on=['UID'], how='left')



used_cols =['city', 'ip', 'nation','device2','geo_code', 
            'district','ip_sub','device1', 'mac1','device_brand',
            'province','device_code']
for col in used_cols:
    stats = user_action.groupby([col])['UID'].nunique().reset_index().rename(columns={'UID': col+'_used_uid_nunique'})
    tmp = user_action.merge(stats, on=[col], how='left')
    gp = tmp.groupby(['UID'])[col+'_used_uid_nunique'].agg({col+'_used_uid_nunique_max': 'max',
                                                            col+'_used_uid_nunique_mean': 'mean',
                                                            col+'_used_uid_nunique_min': 'min',
                                                            col+'_used_uid_nunique_skew': 'skew',
                                                            col+'_used_uid_nunique_mean': 'std'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')
used_cols = ['device_code', 'ip', 'ip_sub', 'mac1', 'device1',
             'device2', 'geo_code']
for col in used_cols:    
    stats = user_action[user_action[col].isnull()==False][['UID', 'actionTimestamp']+[col]].copy()
    stats = stats.sort_values(by=[col, 'actionTimestamp'])
    gp = stats.groupby(col)['actionTimestamp'].agg({col+'_used_user_actionTimestamp_min':'min',
                                                    col+'_used_user_actionTimestamp_max':'max'}).reset_index()
    gp[col+'_used_user_actionTimestamp_timedelta'] = gp[col+'_used_user_actionTimestamp_max'] - gp[col+'_used_user_actionTimestamp_min']
    tmp = user_action.merge(gp, on=[col], how='left')
    gp = tmp.groupby(['UID'])[col+'_used_user_actionTimestamp_timedelta'].agg({col+'_used_user_actionTimestamp_timedelta_max': 'max',
                                                                               col+'_used_user_actionTimestamp_timedelta_min': 'min',
                                                                               col+'_used_user_actionTimestamp_timedelta_mean': 'mean',
                                                                               col+'_used_user_actionTimestamp_timedelta_skew': 'skew',
                                                                               col+'_used_user_actionTimestamp_timedelta_std': 'std'})
    gp.reset_index(inplace=True)
    data = data.merge(gp, on=['UID'], how='left')

used_cols = ['device_code', 'ip', 'ip_sub', 'mac1', 'device1',
             'device2', 'geo_code']
for col in used_cols:    
    stats = user_action[user_action[col].isnull()==False][['UID', 'actionTimestamp']+[col]].copy()
    stats = stats.sort_values(by=[col, 'actionTimestamp'])
    stats['timedelta'] = group_diff_time(stats, [col], 'actionTimestamp', 1, 'timedelta')
    gp = stats.groupby([col])['timedelta'].agg({col+'_used_timedelta_mean': 'mean',
                                                col+'_used_timedelta_skew': 'skew',
                                                col+'_used_timedelta_std': 'std',
                                                col+'_used_timedelta_max': 'max',
                                                col+'_used_timedelta_min': 'min',}).reset_index()
    tmp = user_action.merge(gp, on=[col], how='left')
    columns = [f for f in gp.columns if f not in ['UID']]
    gp = tmp.groupby(['UID'])[columns].agg({'max', 'min', 'mean', 'skew', 'std'})
    gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    data = data.merge(gp, on=['UID'], how='left')


    gp = user_action.groupby(['UID'])['day'].nunique().reset_index().rename(columns={'day': 'uid_action_day_nunique'})
data = data.merge(gp, on=['UID'], how='left')
data['uid_action_day_nunique'].fillna(0, inplace=True)
gp = user_action[user_action['action_type']==1].groupby(['UID'])['day'].nunique().reset_index().rename(columns={'day': 'uid_operation_day_nunique'})
data = data.merge(gp, on=['UID'], how='left')
data['uid_operation_day_nunique'].fillna(0, inplace=True)
gp = user_action[user_action['action_type']==2].groupby(['UID'])['day'].nunique().reset_index().rename(columns={'day': 'uid_trade_day_nunique'})
data = data.merge(gp, on=['UID'], how='left')
data['uid_trade_day_nunique'].fillna(0, inplace=True)
data['uid_trade_day_nunique_ratio'] = data['uid_trade_day_nunique']/(0.01+data['uid_action_day_nunique'])
data['uid_operation_day_nunique_ratio'] = data['uid_operation_day_nunique']/(0.01+data['uid_action_day_nunique'])
data['uid_trade_operation_day_nunique_ratio'] = data['uid_trade_day_nunique']/(0.01+data['uid_operation_day_nunique'])


used_cols = ['device1', 'device2', 'device_code', 'mac1', 'device_brand', 'ip', 'ip_sub', 'geo_code']
for col in used_cols:
    gp = user_action.groupby(['UID', 'day'])[col].nunique().reset_index().rename(columns={col:'uid_action_use_'+col+'_day_nunique'})
    gp = gp.groupby(['UID'])['uid_action_use_'+col+'_day_nunique'].agg({'uid_action_use_'+col+'_day_nunique_mean': 'mean',
                                                                        'uid_action_use_'+col+'_day_nunique_max': 'max',
                                                                        'uid_action_use_'+col+'_day_nunique_min': 'min',
                                                                        'uid_action_use_'+col+'_day_nunique_skew': 'skew',
                                                                        'uid_action_use_'+col+'_day_nunique_std': 'std'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')


    tmp1 = user_action[user_action['action_type']==1].copy()
tmp2 = user_action[user_action['action_type']==2].copy()
gp1 = tmp1.groupby(['UID']).apply(lambda x: set(x['day'].tolist())).reset_index()
gp2 = tmp2.groupby(['UID']).apply(lambda x: set(x['day'].tolist())).reset_index()
gp1.columns = ['UID', 'uid_operation_day_list']
gp2.columns = ['UID', 'uid_trade_day_list']
data = data.merge(gp1, on=['UID'], how='left')
data = data.merge(gp2, on=['UID'], how='left')
data['uid_trade_operation_day_intersection_ratio'] = data.apply(lambda x: calculate_intersection_ratio(x['uid_operation_day_list'], x['uid_trade_day_list']), axis=1)    
del data['uid_operation_day_list'], data['uid_trade_day_list']


gp = user_action.groupby(['UID'])['day'].agg({'uid_action_day_mean': 'mean',
                                              'uid_action_day_max': 'max',
                                              'uid_action_day_min': 'min',
                                              'uid_action_day_sum': 'sum',
                                              'uid_action_day_skew': 'skew',
                                              'uid_action_day_std': 'std'}).reset_index()
data = data.merge(gp, on=['UID'], how='left')



data = w2v_features(data, user_action, ['UID'], 'ip', 10, 'w2v_ip_')
data = w2v_features(data, user_action, ['UID'], 'ip_sub', 10, 'w2v_ip_sub_')
data = w2v_features(data, user_action, ['UID'], 'mac1', 10, 'w2v_mac1_')



stats = user_action.groupby(['UID'])['actionTimestamp'].max().reset_index().rename(columns={'actionTimestamp': 'last_actionTimestamp'})
tmp = user_action.merge(stats, on=['UID'], how='left')
tmp['timedelta'] = tmp['last_actionTimestamp']-tmp['actionTimestamp']
for i in [600,1800,3600,36000,86400,259200,864000]:
    gp = tmp[tmp['timedelta']<=i].copy()
    gp = gp.groupby(['UID', 'action_type']).size().unstack().fillna(0).reset_index()
    gp.columns = ['UID', 'uid_operation_cnt_in_{}s'.format(i), 'uid_trade_cnt_in_{}s'.format(i)]
    gp['uid_action_cnt_in_{}s'.format(i)] = gp['uid_operation_cnt_in_{}s'.format(i)]+gp['uid_trade_cnt_in_{}s'.format(i)]
    gp['uid_trade_ratio_in_{}s'.format(i)] = gp['uid_trade_cnt_in_{}s'.format(i)]/gp['uid_action_cnt_in_{}s'.format(i)]
    gp['uid_trade_operation_ratio_in_{}s'.format(i)] = gp['uid_trade_cnt_in_{}s'.format(i)]/(0.01+gp['uid_operation_cnt_in_{}s'.format(i)])
    data = data.merge(gp, on=['UID'], how='left')



"""
	transaction

	用户各平台交易频次/频率
	用户整体/各平台交易金额
	用户使用的资金来源、交易方式、交易商家、交易金额、账户、转出账户、转入账户、营销活动种类个数
	用户当前交易行为是否使用常用的资金来源、交易方式、交易商家、账户
	用户在黑名单商户交易频次
	用户进行交易的商家\账户的热度,该商家第一次交易时间、最后一次交易时间、交易时间间隔
	用户参与活动交易的频次、金额总和
	用户进行交易的账户、转出账户、转入账户的热度
	商家\交易金额\bal的w2v特征
	用户同一天使用的交易类型、账户、活动、资金源的种类个数
	商户画像（金额、子商户个数， 转入/转出账户个数）
	bal的统计特征
"""

gp = transaction_df[transaction_df['channel'].isin([140, 102, 119])==True].groupby(['UID', 'channel']).size().unstack().reset_index().fillna(0)
gp.columns = ['UID', 'uid_trade_count_on_channel_102', 'uid_trade_count_on_channel_119', 'uid_trade_count_on_channel_140']
data = data.merge(gp, on=['UID'], how='left')
data['uid_trade_ratio_on_channel_102'] = data['uid_trade_count_on_channel_102']/(0.01+data['uid_trade_cnt'])
data['uid_trade_ratio_on_channel_119'] = data['uid_trade_count_on_channel_119']/(0.01+data['uid_trade_cnt'])
data['uid_trade_ratio_on_channel_140'] = data['uid_trade_count_on_channel_140']/(0.01+data['uid_trade_cnt'])


gp = transaction_df.groupby(['UID'])['trans_amt'].agg({'uid_trans_amt_mean': 'mean',
                                                       'uid_trans_amt_sum': 'sum',
                                                       'uid_trans_amt_max': 'max', 
                                                       'uid_trans_amt_min': 'min',
                                                       'uid_trans_amt_std': 'std',
                                                       'uid_trans_amt_skew': 'skew'}).reset_index()
data = data.merge(gp, on=['UID'], how='left')
data['uid_trans_amt_diff'] = data['uid_trans_amt_max']-data['uid_trans_amt_min']


used_cols = ['amt_src1', 'amt_src2', 'trans_type1', 'trans_type2', 'merchant', 'trans_amt', 'acc_id1', 'acc_id2', 'acc_id3', 'market_code', 'market_type']
for col in used_cols:
    gp = transaction_df.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_trade_'+col+'_nunique'})
    data = data.merge(gp, on=['UID'], how='left')


    used_cols = ['amt_src1', 'trans_type1', 'merchant', 'acc_id1']
stats = transaction_df.copy()
for col in used_cols:
    gp = transaction_df.groupby(['UID', col]).size().reset_index().rename(columns={0: 'uid_trade_favor_'+col+'_count'})
    gp = gp.sort_values(by=['UID', 'uid_trade_favor_'+col+'_count'])
    gp = gp.groupby(['UID']).last().reset_index().rename(columns={col: 'uid_trade_favor_'+col})
    stats = stats.merge(gp, on=['UID'], how='left')
    stats['is_trade_favor_'+col] = 0
    stats.loc[stats['uid_trade_favor_'+col]==stats[col], 'is_trade_favor_'+col] = 1
    gp = stats.groupby(['UID'])['is_trade_favor_'+col].agg({'is_trade_favor_'+col+'_count': 'sum',
                                                            'is_trade_favor_'+col+'_mean': 'mean'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')



stats = transaction_df.groupby(['merchant'])['UID'].agg({'merchant_traded_uid_count': 'count',
                                                         'merchant_traded_uid_nunique': 'nunique'}).reset_index()
tmp = transaction_df.merge(stats, on=['merchant'], how='left')
gp = tmp.groupby(['UID'])[['merchant_traded_uid_count', 'merchant_traded_uid_nunique']].agg({'max', 'mean', 'min', 'skew', 'sum', 'std'})
gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')


transaction_df['is_market'] = transaction_df['market_code'].apply(lambda x: 'market' if pd.isna(x)==False else 'nomarket')
gp = transaction_df.groupby(['UID', 'is_market'])['trans_amt'].agg({'count', 'mean'}).unstack().fillna(0)
gp.columns = pd.Index(['uid_trade_on_'+e[1]+"_"+e[0] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')
data['uid_trade_market_count_diff'] = data['uid_trade_on_market_count'] - data['uid_trade_on_nomarket_count']
data['uid_trade_market_mean_diff'] = data['uid_trade_on_market_mean'] - data['uid_trade_on_nomarket_mean']
data['uid_trade_on_market_ratio'] = data['uid_trade_on_market_count'] / (0.01+data['uid_action_cnt'])


gp = transaction_df[transaction_df['acc_id1'].isnull()==False].copy()
gp = gp.groupby(['acc_id1'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id1_traded_uid_nunique'})
stats = transaction_df.merge(gp, on=['acc_id1'], how='left')
gp = stats.groupby(['UID'])['acc_id1_traded_uid_nunique'].agg({'acc_id1_traded_uid_nunique_mean': 'mean',
                                                               'acc_id1_traded_uid_nunique_max': 'max',
                                                               'acc_id1_traded_uid_nunique_min': 'min',
                                                               'acc_id1_traded_uid_nunique_skew': 'skew',
                                                               'acc_id1_traded_uid_nunique_std': 'std'}).reset_index()
gp.fillna(0, inplace=True)
data = data.merge(gp, on=['UID'], how='left')

gp = transaction_df[transaction_df['acc_id2'].isnull()==False].copy()
gp = gp.groupby(['acc_id2'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id2_traded_uid_nunique'})
stats = transaction_df.merge(gp, on=['acc_id2'], how='left')
gp = stats.groupby(['UID'])['acc_id2_traded_uid_nunique'].agg({'acc_id2_traded_uid_nunique_mean': 'mean',
                                                               'acc_id2_traded_uid_nunique_max': 'max',
                                                               'acc_id2_traded_uid_nunique_max': 'min',
                                                               'acc_id2_traded_uid_nunique_skew': 'skew',
                                                               'acc_id2_traded_uid_nunique_std': 'std'}).reset_index()
gp.fillna(0, inplace=True)
data = data.merge(gp, on=['UID'], how='left')

gp = transaction_df[transaction_df['acc_id3'].isnull()==False].copy()
gp = gp.groupby(['acc_id3'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id3_traded_uid_nunique'})
stats = transaction_df.merge(gp, on=['acc_id3'], how='left')
gp = stats.groupby(['UID'])['acc_id3_traded_uid_nunique'].agg({'acc_id3_traded_uid_nunique_mean': 'mean',
                                                               'acc_id3_traded_uid_nunique_max': 'max',
                                                               'acc_id3_traded_uid_nunique_min': 'min',
                                                               'acc_id3_traded_uid_nunique_skew': 'skew',
                                                               'acc_id3_traded_uid_nunique_std': 'std'}).reset_index()
gp.fillna(0, inplace=True)
data = data.merge(gp, on=['UID'], how='left')


data = w2v_features(data, transaction_df, ['UID'], 'merchant', 10, 'w2v_merchant_')
data = w2v_features(data, transaction_df, ['UID'], 'trans_amt', 10, 'w2v_trans_amt_')
data = w2v_features(data, transaction_df, ['UID'], 'bal', 10, 'w2v_bal_')


used_cols = ['merchant', 'acc_id1', 'acc_id2', 'acc_id3']
test_uid = list(data[data['Tag']==0.5]['UID'].unique())
for col in used_cols:
    stats = transaction_df[transaction_df[col].isnull()==False][['UID', 'actionTimestamp']+[col]].copy()
    stats = stats.sort_values(by=[col, 'actionTimestamp'])
    gp = stats.groupby(col)['actionTimestamp'].agg({col+'_traded_user_actionTimestamp_min': 'min',
                                                    col+'_traded_user_actionTimestamp_max': 'max'})
    gp[col+'_traded_user_actionTimestamp_timedelta'] = gp[col+'_traded_user_actionTimestamp_max']-gp[col+'_traded_user_actionTimestamp_min']
    gp.reset_index(inplace=True)
    tmp = transaction_df.merge(gp, on=[col], how='left')
    gp = tmp.groupby(['UID'])[col+'_traded_user_actionTimestamp_timedelta'].agg({col+'_traded_user_actionTimestamp_timedelta_max':'max',
                                                                                 col+'_traded_user_actionTimestamp_timedelta_min': 'min',
                                                                                 col+'_traded_user_actionTimestamp_timedelta_mean': 'mean',
                                                                                 col+'_traded_user_actionTimestamp_timedelta_skew': 'skew',
                                                                                 col+'_traded_user_actionTimestamp_timedelta_std': 'std'})
    gp.reset_index(inplace=True)
    data = data.merge(gp, on=['UID'], how='left')


used_cols = ['amt_src1', 'amt_src2', 'trans_type1', 'trans_type2', 'merchant', 'acc_id1', 'acc_id2', 'acc_id3', 'market_code', 'market_type']
for col in used_cols:
    gp = transaction_df.groupby(['UID', 'day'])[col].nunique().reset_index().rename(columns={col:'uid_trade_use_'+col+'_day_nunique'})
    gp = gp.groupby(['UID'])['uid_trade_use_'+col+'_day_nunique'].agg({'uid_trade_use_'+col+'_day_nunique_mean': 'mean',
                                                                       'uid_trade_use_'+col+'_day_nunique_max': 'max',
                                                                       'uid_trade_use_'+col+'_day_nunique_min': 'min',
                                                                       'uid_trade_use_'+col+'_day_nunique_skew': 'skew',
                                                                       'uid_trade_use_'+col+'_day_nunique_std': 'std'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')



agg = {'code1': ['nunique'],
       'code2': ['nunique'],
       'acc_id1': ['nunique'],
       'acc_id2': ['nunique'],
       'acc_id3': ['nunique'],
       'ip': ['nunique'],
       'geo_code': ['nunique'],
       'trans_amt': ['mean', 'max', 'min', 'std', 'skew'],
       'day': ['mean', 'max', 'min', 'std', 'skew']}
gp = transaction_df.groupby(['merchant']).agg(agg)
gp.columns = pd.Index(['merchant_'+e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
columns = [f for f in gp.columns if f not in ['UID']]
tmp = transaction_df.merge(gp, on=['merchant'], how='left')
gp = tmp.groupby(['UID'])[columns].agg({'max', 'min', 'sum', 'mean', 'std', 'skew'})
gp.columns = pd.Index(['uid_'+e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')
data['uid_trans_amt_mean_vs_merchant_trans_amt_mean'] = data['uid_trans_amt_mean'] / (0.01+data['uid_merchant_trans_amt_mean_mean'])



gp = transaction_df.groupby(['UID'])['bal'].agg({'uid_bal_mean': 'mean',
                                                 'uid_bal_sum': 'sum',
                                                 'uid_bal_max': 'max', 
                                                 'uid_bal_min': 'min',
                                                 'uid_bal_std': 'std',
                                                 'uid_bal_skew': 'skew',
                                                }).reset_index()
data = data.merge(gp, on=['UID'], how='left')
data['uid_bal_diff'] = data['uid_bal_max']-data['uid_bal_min']


"""
operation

用户使用的操作类型、操作系统、版本、wifi的种类个数
用户当前操作是否是最常用用的操作系统、版本、wifi环境、mac2地址
用户第一次使用该设备、该wifi设备的时间、及最后一次使用时间，时间差(mean,max,skew,min,std)
用户同一天使用的操作类型、操作系统、版本、wifi的种类个数
设备、环境使用的最早时间和最晚时间，时间间隔
操作类型、操作版本、wifi、mac2的w2v特征
"""


used_cols = ['mode', 'os', 'version', 'os_version', 'wifi', 'mac2']
for col in used_cols:
    gp = operation_df.groupby(['UID'])[col].nunique().reset_index().rename(columns={col: 'uid_operation_'+col+'_nunique'})
    data = data.merge(gp, on=['UID'], how='left')


used_cols = ['os', 'version', 'os_version', 'wifi', 'mac2']
stats = operation_df.copy()
for col in used_cols:
    gp = operation_df.groupby(['UID', col]).size().reset_index().rename(columns={0: 'uid_operation_favor_'+col+'_count'})
    gp = gp.sort_values(by=['UID', 'uid_operation_favor_'+col+'_count'])
    gp = gp.groupby(['UID']).last().reset_index().rename(columns={col: 'uid_operation_favor_'+col})
    stats = stats.merge(gp, on=['UID'], how='left')
    stats['is_operation_favor_'+col] = 0
    stats.loc[stats['uid_operation_favor_'+col]==stats[col], 'is_operation_favor_'+col] = 1
    gp = stats.groupby(['UID'])['is_operation_favor_'+col].agg({'is_operation_favor_'+col+'_count': 'sum',
                                                                'is_operation_favor_'+col+'_mean': 'mean'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')




used_cols = ['os', 'version', 'os_version', 'wifi', 'mac2']
for col in used_cols:
    tmp = operation_df[['UID', 'actionTimestamp']+[col]].copy()
    gp = tmp.groupby(['UID', col])['actionTimestamp'].agg({col+'_opered_user_actionTimestamp_min': 'min',
                                                           col+'_opered_user_actionTimestamp_max': 'max'})
    gp[col+'_opered_user_actionTimestamp_timedelta'] = gp[col+'_opered_user_actionTimestamp_max']-gp[col+'_opered_user_actionTimestamp_min']
    gp.reset_index(inplace=True)
    gp = gp.groupby(['UID'])[col+'_opered_user_actionTimestamp_timedelta'].agg({col+'_opered_user_actionTimestamp_timedelta_mean': 'mean',
                                                                                col+'_opered_user_actionTimestamp_timedelta_max': 'max', 
                                                                                col+'_opered_user_actionTimestamp_timedelta_min': 'min',
                                                                                col+'_opered_user_actionTimestamp_timedelta_skew': 'skew',
                                                                                col+'_opered_user_actionTimestamp_timedelta_std': 'std'
                                                                                }).reset_index()
    data = data.merge(gp, on=['UID'], how='left')


used_cols = ['os', 'version', 'os_version', 'wifi', 'mac2']
for col in used_cols:
    gp = operation_df.groupby(['UID', 'day'])[col].nunique().reset_index().rename(columns={col:'uid_use_'+col+'_day_nunique'})
    gp = gp.groupby(['UID'])['uid_use_'+col+'_day_nunique'].agg({'uid_use_'+col+'_day_nunique_mean': 'mean',
                                                                 'uid_use_'+col+'_day_nunique_max': 'max',
                                                                 'uid_use_'+col+'_day_nunique_min': 'min',
                                                                 'uid_use_'+col+'_day_nunique_skew': 'skew',
                                                                 'uid_use_'+col+'_day_nunique_std': 'std'}).reset_index()
    data = data.merge(gp, on=['UID'], how='left')



# used_cols = ['os','version', 'os_version', 'wifi', 'mac2']
used_cols = ['os', 'os_version', 'wifi']
for col in used_cols:    
    stats = operation_df[operation_df[col].isnull()==False][['UID', 'actionTimestamp']+[col]].copy()
    stats = stats.sort_values(by=[col, 'actionTimestamp'])
    stats['timedelta'] = group_diff_time(stats, [col], 'actionTimestamp', 1, 'timedelta')
    gp = stats.groupby([col])['timedelta'].agg({col+'_used_timedelta_mean': 'mean',
                                                col+'_used_timedelta_skew': 'skew',
                                                col+'_used_timedelta_std': 'std',
                                                col+'_used_timedelta_max': 'max',
                                                col+'_used_timedelta_min': 'min'}).reset_index()
    tmp = operation_df.merge(gp, on=[col], how='left')
    columns = [f for f in gp.columns if f not in ['UID', col]]
#     gp = tmp.groupby(['UID'])[columns].agg({'max', 'min', 'mean', 'skew', 'std'})
    gp = tmp.groupby(['UID']).agg({'max', 'min', 'mean', 'skew', 'std'})
    gp.columns = pd.Index([e[0]+"_"+e[1] for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    data = data.merge(gp, on=['UID'], how='left')



data = w2v_features(data, operation_df, ['UID'], 'mode', 10, 'w2v_mode_')
data = w2v_features(data, operation_df, ['UID'], 'mac2', 10, 'w2v_mac2_')
data = w2v_features(data, operation_df, ['UID'], 'wifi', 10, 'w2v_wifi_')
data = w2v_features(data, operation_df, ['UID'], 'os', 10, 'w2v_os_')
data = w2v_features(data, operation_df, ['UID'], 'version', 10, 'w2v_version_')
data = w2v_features(data, operation_df, ['UID'], 'os_version', 10, 'w2v_os_version_')




agg = {'channel': ['nunique'],
       'bal': ['mean', 'max', 'min', 'std', 'skew']}
gp = transaction_df.groupby(['merchant']).agg(agg)
gp.columns = pd.Index(['merchant_'+e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
columns = [f for f in gp.columns if f not in ['UID']]
tmp = transaction_df.merge(gp, on=['merchant'], how='left')
gp = tmp.groupby(['UID'])[columns].agg({'max', 'min', 'sum', 'mean', 'std', 'skew'})
gp.columns = pd.Index(['uid_'+e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')




tmp = transaction_df[['UID', 'actionTimestamp']].copy()
tmp = tmp.sort_values(by=['UID', 'actionTimestamp'])
tmp['timedelta'] = group_diff_time(tmp, ['UID'], 'actionTimestamp', 1, 'timedelta')
gp = tmp.groupby(['UID'])['timedelta'].agg({'user_trade_timedelta_mean': 'mean',
                                           'user_trade_timedelta_max': 'max',
                                           'user_trade_timedelta_min': 'min',
                                           'user_trade_timedelta_sum': 'sum',
                                           'user_trade_timedelta_std': 'std',
                                           'user_trade_timedelta_skew': 'skew'}).reset_index()
data = data.merge(gp, on=['UID'], how='left')




def getModeTimeSpan(df_action_of_userid, actiontypeA, actiontypeB, timethred):
    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid) - 1):
        if df_action_of_userid['mode'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTimestamp'].iat[i]
            for j in range(i + 1, len(df_action_of_userid)):
                if df_action_of_userid['mode'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTimestamp'].iat[j]
                if df_action_of_userid['mode'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTimestamp'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
                    break
        i += 1
    return np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0)




tmp1 = operation_df[['UID', 'mode', 'actionTimestamp']].copy()
tmp2 = transaction_df[['UID', 'actionTimestamp']].copy()
tmp2['mode'] = 'trade'
user_action = pd.concat([tmp1, tmp2], axis=0, ignore_index=True)
user_action = user_action.sort_values(by=['UID', 'actionTimestamp'])

userid = user_action['UID'].unique()
timespancount_dict = {'UID': [],
                      'uid_operation_mode1_to_trade_timdelta_count': []}
for uid in userid:
    action_df = user_action[user_action['UID']==uid].copy()
    actiontimespancount = getModeTimeSpan(action_df, 'c8741ce15ceac2a4', 'trade', timethred = 100)
    timespancount_dict['UID'].append(uid)
    timespancount_dict['uid_operation_mode1_to_trade_timdelta_count'].append(actiontimespancount)
timespancount_dict = pd.DataFrame(timespancount_dict)
data = data.merge(timespancount_dict, on=['UID'], how='left')

timespancount_dict = {'UID': [],
                      'uid_operation_mode2_to_trade_timdelta_count': []}
for uid in userid:
    action_df = user_action[user_action['UID']==uid].copy()
    actiontimespancount = getModeTimeSpan(action_df, 'acfaded7e04e7ba0', 'trade', timethred = 100)
    timespancount_dict['UID'].append(uid)
    timespancount_dict['uid_operation_mode2_to_trade_timdelta_count'].append(actiontimespancount)
timespancount_dict = pd.DataFrame(timespancount_dict)
data = data.merge(timespancount_dict, on=['UID'], how='left')



transaction_df = transaction_df.sort_values(by=['UID', 'actionTimestamp'])
agg = {'channel': ['nunique'],
       'acc_id2': ['nunique'],
       'acc_id3': ['nunique'],
       'amt_src1': ['nunique'],
       'amt_src2': ['nunique'],
       'ip': ['nunique'],
       'day': ['mean', 'max', 'min', 'std', 'skew'],
       'trans_amt': ['mean', 'max', 'min', 'std', 'skew'],
       'bal': ['mean', 'max', 'min', 'std', 'skew', 'first', 'last']}
gp = transaction_df.groupby(['acc_id1']).agg(agg)
gp.columns = pd.Index(['acc_id1_'+e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
columns = [f for f in gp.columns if f not in ['UID']]
tmp = transaction_df.merge(gp, on=['acc_id1'], how='left')
gp = tmp.groupby(['UID'])[columns].agg({'max', 'min', 'sum', 'mean', 'std', 'skew'})
gp.columns = pd.Index(['uid_'+e[0]+"_"+e[1] for e in gp.columns.tolist()])
gp.reset_index(inplace=True)
data = data.merge(gp, on=['UID'], how='left')


data.head()

data.to_csv('../cache/features_add.csv', index=False)



















