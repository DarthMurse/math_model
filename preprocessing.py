#数据导入及预处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import  seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')

#数据导入
#aff1.csv即为附件1.csv,aff3.csv为附件3.csv，貌似中文文件名8太行QAQ
df1=pd.read_csv(open('C_aff/2022-C-aff/aff1.csv'),encoding='utf-8',parse_dates=['ds'])
df3=pd.read_csv(open('C_aff/2022-C-aff/aff3.csv'),encoding='utf-8',parse_dates=['ds'])
#标准化处理
df1.value_avg=round((df1.value_avg-df1.value_min)/(df1.value_max-df1.value_min),2)
df3.value_avg=round((df3.value_avg-df3.value_min)/(df3.value_max-df3.value_min),2)

#去除无关属性
df1=df1.drop(labels='value_max',axis=1)
df1=df1.drop(labels='value_min',axis=1)
df1=df1.drop(labels='unit',axis=1)
df3=df3.drop(labels='value_max',axis=1)
df3=df3.drop(labels='value_min',axis=1)
df3=df3.drop(labels='unit',axis=1)

#查看属性
df1.info()
df3.info()

#检查空值
df1.isnull().any()
df3.isnull().any()

#按照'user_id'和'metrics'分组
grouped1=df1.groupby(by=['user_id','metrics'])
grouped3=df3.groupby(by=['user_id','metrics'])
#print(grouped.head())
#print(grouped.groups)

#初始化附件1和附件3的dataframe
#is_loss表示用户是否流失，为1时代表流失，为0时代表用户正常
data1=pd.DataFrame({
    'user_id':[],
    'metrics':[],
    'value_sum':[],
    'duration':[],
    'is_loss':[]
})
data3=pd.DataFrame({
    'user_id':[],
    'metrics':[],
    'value_sum':[],
    'duration':[],
    'is_loss':[]
})

#迭代，添加数据记录
for (key1,key2), group_data in grouped1:
    duration=group_data['ds'].max()-group_data.ds.min()
    summation=group_data['value_avg'].sum()
    data1.loc[len(data1.index)]=[key1,key2,summation,duration,1]

for (key1,key2), group_data in grouped3:
    duration=group_data['ds'].max()-group_data.ds.min()
    summation=group_data['value_avg'].sum()
    data3.loc[len(data3.index)]=[key1,key2,summation,duration,0]

#显示dataframe的信息
data1.info()
data3.info()

#print(data.head())

#合并两附件中的数据
#data即为最终处理所得数据
data=pd.concat([data1,data3])
data.info()
