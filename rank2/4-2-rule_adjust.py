import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


sub = pd.read_csv('../submission/avg_blend_6.csv') # 融合结果
operation_trn = pd.read_csv('../data/operation_train_new.csv')
transaction_trn = pd.read_csv('../data/transaction_train_new.csv')
operation_tst2 = pd.read_csv('../data/test_operation_round2.csv')
transaction_tst2 = pd.read_csv('../data/test_transaction_round2.csv')
tag_trn = pd.read_csv('../data/tag_train_new.csv')


# 规则模拟

# 在119平台交易过的用户都是白用户

train_uid = list(transaction_trn[transaction_trn['channel']==119]['UID'].unique())
white_uid = tag_trn[(tag_trn['Tag']==0)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中白用户%d，占比%.2f'%(len(train_uid), len(white_uid), 1.0*len(train_uid)/len(white_uid)))


# acc_id2

group_df = transaction_trn.groupby(['acc_id2'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id2_uid_nunique'})
stats = transaction_trn.merge(group_df, on=['acc_id2'], how='left')
train_uid = list(stats[stats['acc_id2_uid_nunique']>20]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))



# acc_id3
group_df = transaction_trn.groupby(['acc_id3'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id3_uid_nunique'})
stats = transaction_trn.merge(group_df, on=['acc_id3'], how='left')
train_uid = list(stats[stats['acc_id3_uid_nunique']>10]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))


# transaction:device_code1
group_df = transaction_trn.groupby(['device_code1'])['UID'].nunique().reset_index().rename(columns={'UID': 'device_code1_uid_nunique'})
stats = transaction_trn.merge(group_df, on=['device_code1'], how='left')
train_uid = list(stats[stats['device_code1_uid_nunique']>60]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))


#  transaction:device_code3
group_df = transaction_trn.groupby(['device_code3'])['UID'].nunique().reset_index().rename(columns={'UID': 'device_code3_uid_nunique'})
stats = transaction_trn.merge(group_df, on=['device_code3'], how='left')
train_uid = list(stats[stats['device_code3_uid_nunique']>20]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))


# operation:ip1

group_df = operation_trn.groupby(['ip1'])['UID'].nunique().reset_index().rename(columns={'UID': 'ip1_uid_nunique'})
stats = operation_trn.merge(group_df, on=['ip1'], how='left')
train_uid = list(stats[stats['ip1_uid_nunique']>200]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))


#  transaction:ip1

group_df = transaction_trn.groupby(['ip1'])['UID'].nunique().reset_index().rename(columns={'UID': 'ip1_uid_nunique'})
stats = transaction_trn.merge(group_df, on=['ip1'], how='left')
train_uid = list(stats[stats['ip1_uid_nunique']>200]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))


# operation+transaction:ip1,ip1_sub
user_action_trn = pd.concat([operation_trn[['UID', 'day', 'ip1', 'ip1_sub']].copy(), transaction_trn[['UID', 'day', 'ip1', 'ip1_sub']].copy()], axis=0, ignore_index=True)
user_action_tst2 = pd.concat([operation_tst2[['UID', 'day', 'ip1', 'ip1_sub']].copy(), transaction_tst2[['UID', 'day', 'ip1', 'ip1_sub']].copy()], axis=0, ignore_index=True)


group_df = user_action_trn.groupby(['ip1'])[['UID', 'day']].nunique().reset_index().rename(columns={'UID': 'ip1_uid_nunique',
                                                                                                    'day': 'ip1_day_nunique'})
stats = user_action_trn.merge(group_df, on=['ip1'], how='left')
train_uid = list(stats[(stats['ip1_uid_nunique']>70)&(stats['ip1_day_nunique']<15)]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))



group_df = user_action_trn.groupby(['ip1_sub'])[['UID', 'day']].nunique().reset_index().rename(columns={'UID': 'ip1_sub_uid_nunique',
                                                                                                        'day': 'ip1_sub_day_nunique'})
stats = user_action_trn.merge(group_df, on=['ip1_sub'], how='left')
train_uid = list(stats[(stats['ip1_sub_uid_nunique']>80)&(stats['ip1_sub_day_nunique']<10)]['UID'].unique())
black_uid = tag_trn[(tag_trn['Tag']==1)&(tag_trn['UID'].isin(train_uid)==True)]
print('训练集共找出%d用户，其中黑用户%d，占比%.2f'%(len(train_uid), len(black_uid), 1.0*len(black_uid)/len(train_uid)))


# 规则实现
sub_new = sub.copy()
min_prob = sub_new['Tag'].min()
max_prob = sub_new['Tag'].max()


rule1_uid = list(transaction_tst2[transaction_tst2['channel']==119]['UID'].unique())
sub_new.loc[sub_new['UID'].isin(rule1_uid)==True, 'Tag'] = min_prob

group_df = transaction_tst2.groupby(['acc_id2'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id2_uid_nunique'})
stats = transaction_tst2.merge(group_df, on=['acc_id2'], how='left')
rule2_uid = list(stats[stats['acc_id2_uid_nunique']>20]['UID'].unique())
sub_new.loc[sub_new['UID'].isin(rule2_uid)==True, 'Tag'] = max_prob


group_df = transaction_tst2.groupby(['acc_id3'])['UID'].nunique().reset_index().rename(columns={'UID': 'acc_id3_uid_nunique'})
stats = transaction_tst2.merge(group_df, on=['acc_id3'], how='left')
rule3_uid = list(stats[stats['acc_id3_uid_nunique']>10]['UID'].unique())
sub_new.loc[sub_new['UID'].isin(rule3_uid)==True, 'Tag'] = max_prob



# group_df = transaction_tst2.groupby(['device_code1'])['UID'].nunique().reset_index().rename(columns={'UID': 'device_code1_uid_nunique'})
# stats = transaction_tst2.merge(group_df, on=['device_code1'], how='left')
# rule4_uid = list(stats[stats['device_code1_uid_nunique']>60]['UID'].unique())
# sub_new.loc[sub_new['UID'].isin(rule4_uid)==True, 'Tag'] = max_prob



# group_df = transaction_tst2.groupby(['device_code1'])['UID'].nunique().reset_index().rename(columns={'UID': 'device_code1_uid_nunique'})
# stats = transaction_tst2.merge(group_df, on=['device_code1'], how='left')
# rule4_uid = list(stats[stats['device_code1_uid_nunique']>60]['UID'].unique())
# sub_new.loc[sub_new['UID'].isin(rule4_uid)==True, 'Tag'] = max_prob


# group_df = transaction_tst2.groupby(['device_code3'])['UID'].nunique().reset_index().rename(columns={'UID': 'device_code3_uid_nunique'})
# stats = transaction_tst2.merge(group_df, on=['device_code3'], how='left')
# rule5_uid = list(stats[stats['device_code3_uid_nunique']>20]['UID'].unique())
# sub_new.loc[sub_new['UID'].isin(rule5_uid)==True, 'Tag'] = max_prob


# group_df = operation_tst2.groupby(['ip1'])['UID'].nunique().reset_index().rename(columns={'UID': 'ip1_uid_nunique'})
# stats = operation_tst2.merge(group_df, on=['ip1'], how='left')
# rule6_uid = list(stats[stats['ip1_uid_nunique']>230]['UID'].unique())
# set(rule6_uid).intersection(set(rule1_uid))
# sub_new.loc[sub_new['UID'].isin(rule6_uid)==True, 'Tag'] = max_prob


# group_df = transaction_tst2.groupby(['ip1'])['UID'].nunique().reset_index().rename(columns={'UID': 'ip1_uid_nunique'})
# stats = transaction_tst2.merge(group_df, on=['ip1'], how='left')
# rule7_uid = list(stats[stats['ip1_uid_nunique']>200]['UID'].unique())
# sub_new.loc[sub_new['UID'].isin(rule7_uid)==True, 'Tag'] = max_prob



group_df = user_action_tst2.groupby(['ip1'])[['UID', 'day']].nunique().reset_index().rename(columns={'UID': 'ip1_uid_nunique',
                                                                                                    'day': 'ip1_day_nunique'})
stats = user_action_tst2.merge(group_df, on=['ip1'], how='left')
rule8_uid = list(stats[(stats['ip1_uid_nunique']>70)&(stats['ip1_day_nunique']<15)]['UID'].unique())
sub_new.loc[sub_new['UID'].isin(rule8_uid)==True, 'Tag'] = max_prob




group_df = user_action_tst2.groupby(['ip1_sub'])[['UID', 'day']].nunique().reset_index().rename(columns={'UID': 'ip1_sub_uid_nunique',
                                                                                                         'day': 'ip1_sub_day_nunique'})
stats = user_action_tst2.merge(group_df, on=['ip1_sub'], how='left')
rule9_uid = list(stats[(stats['ip1_sub_uid_nunique']>80)&(stats['ip1_sub_day_nunique']<10)]['UID'].unique())
sub_new.loc[sub_new['UID'].isin(rule9_uid)==True, 'Tag'] = max_prob


sub_new.to_csv('../submission/final3.csv', index=False)




