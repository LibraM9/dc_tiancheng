# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:20:00 2018

@author: 送上年轻的你们
"""

import pandas as pd
INPUT_PATH = '/home/bdr/zst/DataC/data/'
OUT_PATH = '/home/bdr/zst/DataC/data/output1/'
Test_trans = pd.read_csv(INPUT_PATH + 'transaction_round1_new.csv')
Test_tag = pd.read_csv(INPUT_PATH+'sub.csv')  # 测试样本

rule_code = ['5776870b5747e14e' ,'8b3f74a1391b5427' ,'0e90f47392008def' ,'6d55ccc689b910ee' ,'2260d61b622795fb' ,'1f72814f76a984fa' ,'c2e87787a76836e0' ,'4bca6018239c6201' ,'922720f3827ccef8' ,'2b2e7046145d9517' ,'09f911b8dc5dfc32' ,'7cc961258f4dce9c' ,'bc0213f01c5023ac' ,'0316dca8cc63cc17' ,'c988e79f00cc2dc0' ,'d0b1218bae116267' ,'72fac912326004ee' ,'00159b7cc2f1dfc8' ,'49ec5883ba0c1b0e' ,'c9c29fc3d44a1d7b' ,'33ce9c3877281764' ,'e7c929127cdefadb' ,'05bc3e22c112c8c9' ,'5cf4f55246093ccf' ,'6704d8d8d5965303' ,'4df1708c5827264d' ,'6e8b399ffe2d1e80' ,'f65104453e0b1d10' ,'1733ddb502eb3923' ,'a086f47f681ad851' ,'1d4372ca8a38cd1f' ,'29db08e2284ea103' ,'4e286438d39a6bd4' ,'54cb3985d0380ca4' ,'6b64437be7590eb0' ,'89eb97474a6cb3c6' ,'95d506c0e49a492c' ,'c17b47056178e2bb' ,'d36b25a74285bebb']

test_rule_uid = pd.DataFrame(Test_trans[Test_trans['merchant'].isin(rule_code)].UID.unique())
pred_data_rule = Test_tag.merge(test_rule_uid, left_on ='UID',right_on =0, how ='left')
pred_data_rule['Tag'][(pred_data_rule[0]>0)] = 1 # 这个系数还需要调整
pred_data_rule[['UID', 'Tag']].to_csv(OUT_PATH + 'pred_data_rule_1206.csv', index=False)
