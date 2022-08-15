#数据导入及预处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import  seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')

import scipy.stats as stats

def cox_stuart(list_c,debug=False):
    lst=list_c.copy()
    raw_len=len(lst)
    if raw_len%2==1:
        del lst[int((raw_len-1)/2)]
    c=int(len(lst)/2)

    n_pos=n_neg=0
    for i in range(c):
        diff=lst[i+c]-lst[i]
        if diff>0:
            n_pos+=1
        elif diff<0:
            n_neg+=1
        else:
            continue
    num=n_pos+n_neg
    k=min(n_pos,n_neg)
    p_value=2*stats.binom.cdf(k,num,0.5)
    if debug:
        print('fall:%i, rise:%i, p-value:%f'%(n_neg, n_pos, p_value))
    if n_pos>n_neg and p_value<0.05:
        return -1
    elif n_neg>n_pos and p_value<0.05:
        return 1
    else:
        return 0

#数据导入
#aff1.csv即为附件1.csv,aff3.csv为附件3.csv，貌似中文文件名8太行QAQ
df1=pd.read_csv(open('C_aff/2022-C-aff/aff1.csv'),encoding='utf-8',parse_dates=['ds'])
df3=pd.read_csv(open('C_aff/2022-C-aff/aff3.csv'),encoding='utf-8',parse_dates=['ds'])
df4=pd.read_csv(open('C_aff/2022-C-aff/aff4.csv'),encoding='utf-8',parse_dates=['ds'])
#标准化处理
df1.value_avg=round((df1.value_avg-df1.value_min)/(df1.value_max-df1.value_min),2)
df3.value_avg=round((df3.value_avg-df3.value_min)/(df3.value_max-df3.value_min),2)
df4.value_avg=round((df4.value_avg-df4.value_min)/(df4.value_max-df4.value_min),2)

#去除无关属性
df1=df1.drop(labels='value_max',axis=1)
df1=df1.drop(labels='value_min',axis=1)
df1=df1.drop(labels='unit',axis=1)
df3=df3.drop(labels='value_max',axis=1)
df3=df3.drop(labels='value_min',axis=1)
df3=df3.drop(labels='unit',axis=1)
df4=df4.drop(labels='value_max',axis=1)
df4=df4.drop(labels='value_min',axis=1)
df4=df4.drop(labels='unit',axis=1)

#查看属性
#df1.info()
#df3.info()

#检查空值
df1.isnull().any()
df3.isnull().any()

#按照'user_id'和'metrics'分组
grouped1=df1.groupby(by=['user_id','metrics'])
grouped3=df3.groupby(by=['user_id','metrics'])
grouped4=df4.groupby(by=['user_id','metrics'])
#print(grouped.head())
#print(grouped.groups)

#初始化附件1和附件3的dataframe
#is_loss表示用户是否流失，为1时代表流失，为0时代表用户正常
data1=pd.DataFrame({
    'user_id':[],
    'metrics':[],
    'trend':[],
    'is_loss':[]
})
data3=pd.DataFrame({
    'user_id':[],
    'metrics':[],
    'trend':[],
    'is_loss':[]
})
data4=pd.DataFrame({
    'user_id':[],
    'metrics':[],
    'trend':[],
    'is_loss':[]
})

#迭代，添加数据记录
for (key1,key2), group_data in grouped1:
    trend=cox_stuart(list(group_data['value_avg']))
    data1.loc[len(data1.index)]=[key1,key2,trend,1]

for (key1,key2), group_data in grouped3:
    trend=cox_stuart(list(group_data['value_avg']))
    data3.loc[len(data3.index)]=[key1,key2,trend,0]

for (key1,key2), group_data in grouped4:
    trend=cox_stuart(list(group_data['value_avg']))
    data4.loc[len(data4.index)]=[key1,key2,trend,0]
#显示dataframe的信息
#data1.info()
#data3.info()

#print(data.head())

#合并两附件中的数据
#data即为最终处理所得数据
data=pd.concat([data1,data3])
data.info()

feature_vec=['1,1,1', '10,1,11', '10,1,12', '10,1,18', '13,1,7', '4,1,2', '4,1,3', '4,2,2', '4,2,3']

for key in feature_vec:
    data[key]=0
    data4[key]=0
    for index,row in data.iterrows():
        if row['metrics'] == key:
            row[key]=row['trend']
    for index, row in data4.iterrows():
        if row['metrics'] == key:
            row[key] = row['trend']

data=data.drop(labels='metrics',axis=1)
data=data.drop(labels='trend',axis=1)
data4=data4.drop(labels='metrics',axis=1)
data4=data4.drop(labels='trend',axis=1)
grouped=data.groupby('user_id')
data_=data.drop(index=data.index)
grouped4=data4.groupby('user_id')
data4_=data4.drop(index=data4.index)
for group_name, group_data in grouped:
    is_loss=group_data.is_loss.iloc[0]
    data_.loc[len(data_.index)]=[group_name,is_loss,
                                 group_data['1,1,1'].sum(),group_data['10,1,11'].sum(),
                                 group_data['10,1,12'].sum(),group_data['10,1,18'].sum(),
                                 group_data['13,1,7'].sum(),group_data['4,1,2'].sum(),
                                 group_data['4,1,3'].sum(),group_data['4,2,2'].sum(),
                                 group_data['4,2,3'].sum()]

for group_name, group_data in grouped4:
    is_loss=group_data.is_loss.iloc[0]
    data4_.loc[len(data4_.index)]=[group_name,is_loss,
                                 group_data['1,1,1'].sum(),group_data['10,1,11'].sum(),
                                 group_data['10,1,12'].sum(),group_data['10,1,18'].sum(),
                                 group_data['13,1,7'].sum(),group_data['4,1,2'].sum(),
                                 group_data['4,1,3'].sum(),group_data['4,2,2'].sum(),
                                 group_data['4,2,3'].sum()]

data_.info()

x=data_.loc[:,['1,1,1', '10,1,11', '10,1,12', '10,1,18', '13,1,7', '4,1,2', '4,1,3', '4,2,2', '4,2,3']]
x=np.array(x)

x4=data4_.loc[:,['1,1,1', '10,1,11', '10,1,12', '10,1,18', '13,1,7', '4,1,2', '4,1,3', '4,2,2', '4,2,3']]
x4=np.array(x4)

y=data_.is_loss
y=np.array(y)
y = y[:, np.newaxis]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini',
                                    splitter='best',
                                    max_depth=5,
                                    min_samples_split=5,
                                    min_samples_leaf=3
                                    )
clf = clf.fit(x_train,y_train)

train_score = clf.score(x_train,y_train) # 训练集的评分
test_score = clf.score(x_test,y_test)   # 测试集的评分

print('train_score:{0},test_score:{1}'.format(train_score,test_score))

# 模型的参数调优--max_depth
# 创建一个函数，使用不同的深度来训练模型，并计算评分数据
#def cv_score(d):
#    clf2 = tree.DecisionTreeClassifier(max_depth=d)
#    clf2 = clf2.fit(x_train,y_train)
#    tr_score = clf2.score(x_train,y_train)
#    cv_score = clf2.score(x_test,y_test)
#    return (tr_score, cv_score)
# 构造参数范围，在这个范围内构造模型并计算评分
#depths = range(2,15)
#scores = [cv_score(d) for d in depths]
#tr_scores = [s[0] for s in scores]
#cv_scores = [s[1] for s in scores]
# 找出交叉验证数据集最高评分的那个索引
#best_score_index = np.argmax(cv_scores)
#best_score = cv_scores[best_score_index]
#best_param = depths[best_score_index]
#print('best_param : {0},best_score: {1}'.format(best_param,best_score))

#plt.figure(figsize = (4,2),dpi=150)
#plt.grid()
#plt.xlabel('max_depth')
#plt.ylabel('best_score')
#plt.plot(depths, cv_scores,'.g-',label = 'cross_validation scores')
#plt.plot(depths,tr_scores,'.r--',label = 'train scores')
#plt.legend()

y_pred=clf.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

y_pred4=clf.predict(x4)
user=data4_.loc[:,'user_id']
user=np.array(user)
user=np.concatenate((user,y_pred4),axis=0)
print(user)