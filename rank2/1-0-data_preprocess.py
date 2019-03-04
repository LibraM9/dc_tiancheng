import pandas as pd
import numpy as np
import warnings
import datetime
import time
from dateutil.parser import parse
warnings.filterwarnings('ignore')


input_path = 'F:/数据集/1207甜橙金融/data/'
output_path = 'F:/数据集处理/甜橙/'

# nrows = 100000

f = open(input_path+'operation_train_new.csv', encoding='utf-8')
operation_trn = pd.read_csv(f)
f = open(input_path+'transaction_train_new.csv', encoding='utf-8')
transaction_trn = pd.read_csv(f)
f = open(input_path+'operation_round1_new.csv', encoding='utf-8')
operation_tst1 = pd.read_csv(f)
f = open(input_path+'transaction_round1_new.csv', encoding='utf-8')
transaction_tst1 = pd.read_csv(f)
f = open(input_path+'test_operation_round2.csv', encoding='utf-8')
operation_tst2 = pd.read_csv(f)
f = open(input_path+'test_transaction_round2.csv', encoding='utf-8')
transaction_tst2 = pd.read_csv(f)
f = open(input_path+'GeoDetails.csv', encoding='utf-8')
geo_hash = pd.read_csv(f)
del geo_hash['street'], geo_hash['street_number']

operation_trn['day_new'] = operation_trn['day'].values
transaction_trn['day_new'] = transaction_trn['day'].values
operation_tst1['day_new'] = operation_tst1['day'].values+30
transaction_tst1['day_new'] = transaction_tst1['day'].values+30
operation_tst2['day_new'] = operation_tst2['day'].values+61
transaction_tst2['day_new'] = transaction_tst2['day'].values+61

oper_data = pd.concat([operation_trn, operation_tst1, operation_tst2], axis=0, ignore_index=True)
trans_data = pd.concat([transaction_trn, transaction_tst1, transaction_tst2], axis=0, ignore_index=True)


# 工具函数
brand_dict = {
'huawei': ['R7C','C8817D','CHE1-CL20','SCL-CL00','CHE1-CL10','MHA-Al00','HUAWEI','Huawei','HUAWEI ALE-CL00','HUAWEI NXT-CL00','HUAWEIMLA-AL10','HUAWEI MLA-AL10','HUAWEI MLA-AL00','huawei','HUAWEI TIT-AL00','honor','Honor','HONOR','GEM-703L','CHM-CL00','VKY-AL00','MHA-AL00','EVA-TL00','PIC-AL00','PRA-AL00','KNT-AL20','CUN-AL00','KNT-UL10','LDN-AL00','NCE-AL00','DIG-TL10','FLA-AL10','EML-AL00','WAS-AL00','EVA-AL10','BLN-TL10','RNE-AL00','BLA-AL00','HWI-AL00','DLI-AL10','BND-AL00','SLA-AL00','FRD-AL10','KIW-TL00H','MYA-AL10','EDI-AL10','NEM-AL10','BLN-AL20','LND-AL30','EDI-AL10','EVA-AL00','CLT-AL01','TRT-TL10','DUK-AL20','DIG-AL00','TRT-AL00','JMM-AL00','BLN-AL40','VIE-AL10','LON-AL00','CHM-TL00H','AUM-AL20','BND-AL10','PRA-AL00X','CLT-AL00','BAC-AL00','STF-AL10','FRD-AL00','ATH-AL00','NCE-AL10','FLA-AL20','ANE-AL00','SCL-TL00H','COL-AL10','SLA-TL10','VTR-TL00','PLK-AL10','KIW-UL00','CAM-AL00','ALP-TL00','NEM-TL00','DUK-TL30','KIW-CL00','Che1-CL10','CHE-TL00H','FIG-TL10','TRT-TL10A','LLD-AL00','H60-L03','BKL-TL10','STF-TL10','ATU-TL10','PAR-AL00','LDN-TL20','NTS-AL00','CUN-TL00','LND-TL40','CHE-TL00','PE-TL20','LDN-TL00','MYA-TL10','Che2-UL00','Che2-TL00','ANE-TL00','LLD-AL20','NEM-TL00H','ATH-TL00H','FIG-TL00','Che2-TL00M','CAM-UL00','ALE-TL00','ATU-AL10','PE-TL10','PRA-TL10','PE-TL00M','PE-CL00','SCL-TL00','PLK-TL00','LND-AL40','BLN-AL30','HWI-TL00','NCE-TL10','EML-TL00','FLA-TL10','AUM-AL00','PLK-CL00','SCL-AL00','BKL-AL00','CHM-UL00','ALP-AL00','BAC-TL00','STF-AL00','TRT-AL00A','BKL-AL20','KIW-TL00','ALE-UL00','CAM-TL00','BLN-AL10','VTR-AL00','LLD-AL10','VKY-TL00','KNT-AL10','DLI-TL20','COR-AL00','KIW-AL10','BTV-DL09','PIC-TL00','LLD-AL30','LLD-TL10','CHM-TL00','NEM-UL10','BND-TL10','FIG-AL10','JMM-AL10','COR-AL10','BLA-TL00','EVA-DL00','PLK-UL00','WAS-TL10','CAM-TL00H','FIG-AL00','H60-L01','v10','Che1-CL20','PLE-703L','H60-L02','PLK-TL01H','MHA-TL00','LDN-AL20','JMM-TL10'],
'apple': ['8 ΡLUS','IPHONE','iPhone9', 'iphone9', 'iPhone8', 'APPLE', 'Apple', 'iphone8', 'iphone7', 'iPhone7', 'iPhone9s', 'iphone9s', 'iPhoneX', 'iphonex', 'iphone', 'iPhone', '苹果六', '苹果'],
'xiaomi': ['HM 1S','2014812','Xiaomi','xiaomi','XIAOMI','MI 6','2014813','MI MAX 2','Redmi 5 Plus','Redmi 5A','MI','Redmi Note 5A','Redmi Note 4','Redmi 4','Redmi 3S','Redmi Note 5','Redmi Note 4X','MI NOTE Pro','MI 4S','MI 3W','Redmi S2','Redmi%20Note%203','MI 3','Mi Note 3','MI 5','MI 5X','MI 6X','MI 5s Plus','Redmi 4X','Redmi 3X','MI 8 SE','Redmi 5','Redmi 6 Pro','Redmi 3','Redmi Note 2','Redmi Note 3','MI 8','MIX 2','Redmi 4A','Mi-4c','MI MAX','HM NOTE 1S','MIX 2S','MI%205','HM 2A','Redmi%20Note%204','REDMI','RedmiNote4X','MI 5s','MI 4LTE','SKR-A0','MI 5C','MIX','M2 E','Redmi Pro','RedmiNote3','MI NOTE LTE','HM NOTE 1LTE','MINOTELTE','2014501','2014811'],
'oppo': ['F5','PACM00','OPPO','OPPO A59s','OPPO A57','OPPO A79k','OPPO R9s Plus','OPPO R11 Plus','oppo','OPPO A37t','OPPO R11st','OPPO R11','OPPO R9s','OPPO R11t','3007','R8205','A31c','OPPO A59m','OPPO R9m','OPPO A73t','OPPO A33','PADM00','PACM00','PAAT00','PADT00','PACT00','A51','PAAM00','R7Plus','R7Plusm','PBAM00','A31','R7PLUSM','R8207','R8107','R7c','PAFM00'],
'vivo': ['vivo','vivo Y79A','vivo Y55','vivo Y79','vivo V3Max A','vivo X20A','VIVO','vivo Y66L','vivo Y67L','vivo Y31A','vivo Y67A','vivo Xplay3S','vivo Y71','vivo X9i','vivo Y66','vivo X9','vivo X5S L','vivo Xplay5A','vivo X7','vivo Y66i','vivo X21i A','vivo Y51','vivo X9L','vivo Y55A','Vivo','BBK','X20','V9S'],
'meizu': ['M571C','M5 NOTE','Meizu','meizu','MEIZU M6','Meizu S6','MX5','MX4 Pro','M5 Note','m1','M1','M6','MX4','M3','m3 note','m2','MX6','M6 Note','m1%20metal','Mi Note 2','m3','M3 Max','m1 metal','M2','m1 note','PRO 7-S','M621C','M3X','M1 E','PRO 5','M5s','PRO 6','PRO 6s','PRO 7-H','m2 note','PRO 7 Plus','Y685C','U20','M578C'],
'samxing': ['SM-G9350','SM-G955F','samsung','Samsung','SM-G9550','SM-C5000','SM-C7000','SM-G9500','SM-G5700','GT-N7100','SM-N9200SM-G9350','SM-G6100','SM-J5008','SM-G9250','SM-A5009','SM-G9200','SM-A7000','SM-N9100','SM-N9006','SM-A9000','SM-E7000','SM-N9009','SM-J3109','SM-A8000','SM-N9008V','SM-G9300','SM-C7100','SM-G9008W','SM-N900','SM-J3300','SM-W2016','SM-N9002','SM-A5100','SM-G9208','SM-J7008','SM-A7009','SM-A5000','SM-J5108','SM-J3119','SM-G5500','SM-C7010','SM-G9280','SM-G9209','SM-G6000','SM-C9000','SM-C5010','SM-G9008V','SM-G9600','SM-A9100','SM-G9650','S9','SM-N9500','SM-A7100','SAMSUNG'],
'lianxiang': ['LENOVO','Lenovo','lenovo','ZUK','ZUK Z2121','ZUK Z2131'],
'nojiya': ['NOAIN','Nokia','Nokia X6','TA-1041','TA-1000','TA-1054'],
'honglajiao': ['xiaolajiao','LA-S35','LA-S33','红辣椒','honglajiao','YUSUN','LA-S31','LA-X7'],
'yueshi': ['letv','Letv','x600','LeMobile','Le','Le X620','Le X528','LeEco','LEX720','LETV','Letv X501','Letv X500','LEX626','lemobile','LeX620','LeTV','Le X520','LEX622','Le X820','leeco','LE','X900'],
'htc': ['HTC','htc'],
'bailifeng': ['Blephone','CBL','LEPHONE'],
'tcl': ['TCL','TCT'],
'jinli': ['M5','GiONEE','GIONEE','GN5001S','GIONEE S10L','GIONEE S10BL','GN8003L','GIONEE F6L','gionee','GN8001','GIONEE S10CL','F105','GN9011','GN5005','GN9012','GIONEE S10','F106L','GN3003','Gionee','GIONEE F6','F100','GN8003','GN8002S','GIONEE M7','GN5002','GN5003','GIONEE GN5007','GN9006','GN9010','F100L','GIONEE S10C','GIONEE M7L','GIONEE S10B','GIONEE GN5007L','GIONEE F109L','GN5005L','F100S','F106','GN3001','W909'],
'chuizi': ['OD103','smartisan','SMARTISAN','YQ601','T1','OS105','OC105','OD105'],
'_360': ['360','1603-A03','1707-A01','1807-A01','1607-A01','1605-A01','1505-A02','1801-A01'],
'meitu': ['Meitu','MP1602','MP1503'],
'zhongxin': ['ZTE','zte','S36','ZTE BV0730','ZTE A2017','ZTEtech'],
'wopufeng': ['WellPhone','wellphone','WEllPHONE'],
'kupai': ['8298-A01','Coolpad','Yulong','coolpad','C106-9','C106','COOLPAD'],
'qiku': ['qiku','QiKU','Qiku','1505-A01'],
'nubiya': ['nubia','NX523J_V1','NX511J_V3','NX529J','NX575J','NX569H','NX563J','NX569J','NX510J','NX513J','NX531J','NX549J','NX573J','NX511J','NX512J','NX541J','NX508J'],
'yijia': ['OnePlus','ONEPLUS','ONEPLUS A3010','ONEPLUS A5000','ONEPLUS A6000','ONEPLUS A5010'],
'duowei': ['DOOV','doov','Doov'],
'yufeilai': ['YUFLY','YU FLY','YU FLY F9'],
'zhizunbao': ['Best sonny','BESTSONNY','Best+sonny','Best+Sonny','Best-sonny','Best_sonny'],
'suoni': ['Sony Ericsson','Sony','sony'],
'zhongguoyidong': ['CMDC','CMCC','M651CY','M A5','M653','China Mobile'],
'hmdglobal': ['HMD Global','HMD+Global+Oy','HMD-Global','HMD-Global-Oy','HMD Global Oy','HMD+Global'],
'tianyu': ['K-Touch','K-TOUCH','%E5%A4%A9%E8%AF%AD','4G%2B','天语','%E7%BA%A2%E8%BE%A3%E6%A4%92'],
'jinyilai': ['Lovme-T19','Lovme','Lovme-T26'],
'aoluosi': ['ALOES','G19','aloes'],
'yuping': ['Y01(YP01)','YEPEN','i7S(YP7S)'],
'yonglongtong': ['YLTphone','YLT'],
'geli': ['Gree','GREE','GELI'],
'tangguo': ['SUGAR','SOAP'],
'shijitianyuan': ['ctyon','shijitianyuan'],
'bodao': ['Bird','bird','BIRD'],
'aole': ['h7571','AOLE'],
'vvetime': ['VVETIME','VVETRAIN'],
'baihe': ['BIHEE','BIHEE A7','zhongxinbaihe'],
'shouyun': ['SHOWN_P1','SHOWN'],
'yousi': ['Uniscope','uniscope','US688'],
'weimi': ['weiimi','weimi'],
'hengyufeng': ['HEYUF','HYF'],
'zhanxun': ['sprd','SPRD','sprd','Spreadtrum'],
'konka':['KONKA', 'konka'],
'guangxin':['KINGSUN-F70','KINGSUN-F9'],
'haixin':['HISENSE']}

def clear_device_code(code1, code2, code3):
    if pd.isna(code1):
        code1 = ''
    if pd.isna(code2):
        code2 = ''
    if pd.isna(code3):
        code3 = ''
    return code1+code2+code3

def device2_clear(x):
    x = str(x)
    y = ''
    if x == 'nan':
        y = '-1'
    else:
        for keys, items in brand_dict.items():
            for i in items:
                if x.find(i) != -1:
                    y = keys
        if len(y) == 0:
            y = 'others'
    return y

# 1:ios 2:android 3:unknown
def check_device_type(code1, code2, code3):
    if (pd.isna(code1)==False)|(pd.isna(code2)==False):
        return 1
    if pd.isna(code3)==False:
        return 2
    else:
        return -1
    
def fill_nans(df):
    for col in df.columns:
        if df[col].isnull().sum()>0:
            if df[col].dtypes == 'object':
                df[col].fillna('-1', inplace=True)
            else:
                df[col].fillna(-1, inplace=True)
    return df

def is_china(x):
    if x=='中国':
        return 1
    if pd.isna(x)==True:
        return -1
    else:
        return 0
# 1:computer 2:phone 3:both
def check_oper_env(ip1, ip2):
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==False):
        return 1
    if (pd.isna(ip1)==False)&(pd.isna(ip2)==True):
        return 2
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==True):
        return -1
    else:
        return 3
    
def ip_clear(ip1, ip2):
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==False):
        return ip2
    if (pd.isna(ip1)==False)&(pd.isna(ip2)==True):
        return ip1
    if (pd.isna(ip1)==True)&(pd.isna(ip2)==True):
        return np.nan
    else:
        return ip1
    
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + datetime.timedelta(days = days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date

def timestamp_transform(a):
    #将python的datetime转换为unix时间戳
    timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
    timeStamp = float(time.mktime(timeArray))
    #将unix时间戳转换为python  的datetime
    return timeStamp



# 数据预处理
# 定义完整时间
oper_data['actionTime'] = oper_data['day_new'].apply(lambda x: date_add_days('2018-08-31', x))
trans_data['actionTime'] = trans_data['day_new'].apply(lambda x: date_add_days('2018-08-31', x))
oper_data['actionTime'] = oper_data['actionTime'].map(str)+' '+oper_data['time'].map(str)
trans_data['actionTime'] = trans_data['actionTime'].map(str)+' '+trans_data['time'].map(str)
oper_data['actionTimestamp'] = oper_data['actionTime'].apply(lambda x: timestamp_transform(x))
trans_data['actionTimestamp'] = trans_data['actionTime'].apply(lambda x: timestamp_transform(x))
# 重新定义device_code
oper_data['device_code'] = oper_data.apply(lambda x: clear_device_code(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
trans_data['device_code'] = trans_data.apply(lambda x: clear_device_code(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
oper_data['device_code'].replace({'': np.nan}, inplace=True)
trans_data['device_code'].replace({'': np.nan}, inplace=True)
# 判断苹果/安卓机
oper_data['device_type'] = oper_data.apply(lambda x: check_device_type(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
trans_data['device_type'] = trans_data.apply(lambda x: check_device_type(x['device_code1'], x['device_code2'], x['device_code3']), axis=1)
del oper_data['device_code1'], oper_data['device_code2'], oper_data['device_code3']
del trans_data['device_code1'], trans_data['device_code2'], trans_data['device_code3']
# device2提取出品牌
oper_data['device_brand'] = oper_data['device2'].apply(lambda x: device2_clear(x))
trans_data['device_brand'] = trans_data['device2'].apply(lambda x: device2_clear(x))
# 匹配地理信息
oper_data = oper_data.merge(geo_hash, on=['geo_code'], how='left')
trans_data = trans_data.merge(geo_hash, on=['geo_code'], how='left')
# 设备版本信息
oper_data['os_version'] = oper_data['os'].map(str)+'_'+oper_data['version'].map(str)
# 时间信息
oper_data['hour'] = oper_data['time'].str[0:2].astype(int)
trans_data['hour'] = trans_data['time'].str[0:2].astype(int)
# 是否是wifi环境
oper_data['is_wifi_env'] = oper_data['wifi'].apply(lambda x: 1 if pd.isna(x)==True else 0)
# 是否中国地区
oper_data['is_china'] = oper_data['nation'].apply(lambda x: is_china(x))
trans_data['is_china'] = trans_data['nation'].apply(lambda x: is_china(x))
# 是否电脑
oper_data['oper_ip_env'] = oper_data.apply(lambda x: check_oper_env(x['ip1'], x['ip2']), axis=1)
# 重新定义ip
oper_data['ip'] = oper_data.apply(lambda x: ip_clear(x['ip1'], x['ip2']), axis=1)
oper_data['ip_sub'] = oper_data.apply(lambda x: ip_clear(x['ip1_sub'], x['ip2_sub']), axis=1)
del oper_data['ip1'], oper_data['ip1_sub'], oper_data['ip2'], oper_data['ip2_sub']
trans_data = trans_data.rename(columns={'ip1': 'ip',
                                        'ip1_sub': 'ip_sub'})
# 设备信息缺失程度
device_cols1 = ['os', 'version', 'device1', 'device2', 'device_code', 'mac1']
device_cols2 = ['device1', 'device2', 'device_code', 'mac1']
oper_data['device_miss_cnt'] = oper_data[device_cols1].isnull().sum(axis=1)
trans_data['device_miss_cnt'] = trans_data[device_cols2].isnull().sum(axis=1)
# 环境信息缺失程度
env_cols1 = ['wifi', 'ip', 'ip_sub', 'mac2', 'geo_code']
env_cols2 = ['ip', 'ip_sub', 'geo_code']
oper_data['env_miss_cnt'] = oper_data[env_cols1].isnull().sum(axis=1)
trans_data['env_miss_cnt'] = trans_data[env_cols2].isnull().sum(axis=1)
# # 缺失值填补
# oper_data = fill_nans(oper_data)
# trans_data = fill_nans(trans_data)
oper_data['success'].fillna(-1, inplace=True)


del oper_data['day_new'], trans_data['day_new']

oper_data.to_csv(output_path+'operation_pre.csv', encoding='gbk', index=False)
trans_data.to_csv(output_path+'transaction_pre.csv', encoding='gbk', index=False)