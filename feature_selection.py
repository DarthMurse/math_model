#数据导入及预处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import  seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')

#数据导入
df1=pd.read_csv(open('C_aff/2022-C-aff/aff1.csv'),encoding='utf-8',parse_dates=['ds'])
#标准化处理
df1.value_avg=round((df1.value_avg-df1.value_min)/(df1.value_max-df1.value_min),2)

#去除无关属性
df1=df1.drop(labels='value_max',axis=1)
df1=df1.drop(labels='value_min',axis=1)
df1=df1.drop(labels='unit',axis=1)

#查看属性
#df1.info()

#检查空值
#df1.isnull().any()

#按照'user_id'和'metrics'分组
grouped1=df1.groupby(by=['user_id','metrics'])
#print(grouped.head())
#print(grouped.groups)

#初始化附件1和附件3的dataframe
#is_loss表示用户是否流失，为1时代表流失，为0时代表用户正常
data1=pd.DataFrame({
    'user_id':[],
    'metrics':[],
    'value_avg':[],
    'duration':[],
    'is_loss':[]
})

#迭代，添加数据记录
for (key1,key2), group_data in grouped1:
    duration=group_data['ds'].max()-group_data.ds.min()
    summation=group_data['value_avg'].sum()
    if duration.days!=0:
        summation=summation/duration.days
    data1.loc[len(data1.index)]=[key1,key2,summation,duration,1]

#显示dataframe的信息
#data1.info()

grouped=data1.groupby('metrics')
var_thresh=1
cnt_thresh=50
list=[]
for group_name, group_data in grouped:
    if group_data['value_avg'].count() > cnt_thresh:
        var=group_data['value_avg'].var()
        if var>var_thresh:
            list.append(group_name)

print(len(list))
print(list)