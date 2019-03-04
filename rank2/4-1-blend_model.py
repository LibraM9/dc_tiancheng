import pandas as pd


# 加权融合
sub1 = pd.read_csv('../submission/lgb_basesub.csv')# 线上0.479+
sub2 = pd.read_csv('../submission/lgb_embeding_eepWalk.csv')# 线上0.37+
sub3 = pd.read_csv('../submission/lgb_embeding_pca.csv')# 线上0.37+

sub1 = sub1.rename(columns={'Tag': 'Tag1'})
sub2 = sub2.rename(columns={'Tag': 'Tag2'})
sub3 = sub3.rename(columns={'Tag': 'Tag3'})



sub = sub1.merge(sub2, on=['UID'], how='left')
sub = sub.merge(sub3, on=['UID'], how='left')
sub['Tag'] = 0.50*sub['Tag1']+0.26*sub['Tag2']+0.24*sub['Tag3']

# 输出结果
sub[['UID', 'Tag']].to_csv('../submission/avg_blend_6.csv', index=False)