import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in csv files
dataset1 = pd.read_csv('data/1.csv')
dataset3 = pd.read_csv('data/3.csv')
dataset2 = pd.read_csv('data/2.csv')
dataset4 = pd.read_csv('data/4.csv')

# Find earliest and latest time
earliest1 = min(dataset1['ds'])
latest1 = max(dataset1['ds'])
earliest3 = min(dataset3['ds'])
latest3 = max(dataset3['ds'])
earliest4 = min(dataset4['ds'])
latest4 = max(dataset4['ds'])
earliest = min(earliest4, earliest3)
latest = max(latest4, latest3)

# Count time point
time_point = pd.date_range(earliest, latest, freq='3D')
count = len(time_point)

def extract_single_seq(user_name, metric, dataset):
    user_data = dataset.loc[dataset['user_id'] == user_name]
    metric_data = user_data[user_data['metrics'].str.startswith(str(metric)+',')]

    # Defining the starting day and ending day
    start_date = min(user_data['ds'])
    end_date = max(user_data['ds'])
    
    date = []
    series = []

    for day in pd.date_range(start_date, end_date, freq='3D'):
        date.append(day)
        series.append(metric_data.loc[pd.to_datetime(metric_data['ds']) == day]['value_avg'].sum())
    new_sery = pd.Series(series, index=date)
    return new_sery

def user_metric_to_array(user_name, dataset, metric, total=False):
    pd_sery = extract_single_seq(user_name, metric, dataset)
    values = pd_sery.values
    date = pd_sery.index
    start_point = (date[0] - time_point[0]) // pd.Timedelta('3 days')
    values_length = len(values)

    if total:
        result = np.zeros(count)
        result[start_point:start_point+values_length] = np.asarray(values)
    else:
        result = np.asarray(values)
    return result

def user_to_array(user_name, dataset, total=False):
    length = len(user_metric_to_array(user_name, dataset, 1))
    if total:
        result = np.zeros([15, count])
    else:
        result = np.zeros([15, length])
    for i in range(15):
        result[i] = user_metric_to_array(user_name, dataset, i+1, total)

    return result, length

def user_resource(user_name, dataset, total=False):
    metrics, length = user_to_array(user_name, dataset, total)
    result = np.zeros([3, metrics.shape[1]])
    result[0] = metrics[2] + metrics[7] + metrics[9] + metrics[12]  # computation
    result[1] = metrics[0] + metrics[1] + metrics[4] + metrics[10] + metrics[11] + metrics[14]  # storage
    result[2] = metrics[3] + metrics[5] + metrics[6] + metrics[13]
    return result, length

def user_drop_date(user_name):
    sery = dataset2[dataset2['User_id'] == user_name].iloc[0]
    date = sery.dropna()
    date = date[1:]
    return date

def left_time(user_name, dataset):
    user_data = dataset.loc[dataset['user_id'] == user_name]

    # Defining the starting day and ending day
    start_date = min(user_data['ds'])
    end_date = max(user_data['ds'])
    warning_date = user_drop_date(user_name)[-1]
    
    result = (pd.to_datetime(end_date) - pd.to_datetime(warning_date)) // pd.Timedelta('3 days')
    return result

def get_user_ids(dataset):
    user_ids = dataset['user_id']
    result = [user_ids[0]]
    for term in user_ids:
        if term != result[-1]:
            result.append(term)
    return result

def preprocessing(dataset, is_dataset1):
    user_ids = get_user_ids(dataset)
    size = len(user_ids)
    result = np.zeros([size, count * 3])

    i = 0
    for user_name in user_ids:
        user_rsc, _ = user_resource(user_name, dataset, True)
        result[i] = np.concatenate(user_rsc)
        i += 1
        if i % 10 == 0:
            print(f"{i} users processed")

    '''
    label = np.zeros([size, 3])
    if is_dataset1:
        label[:, 0] = np.ones([size])
        for i in range(size):
            label[i, 2] = left_time(user_ids[i], dataset) / 180

    if not is_dataset1:
        label[:, 1] = np.ones([size])
    '''
    
    return result

if __name__ == '__main__':
    '''
    data1, label1 = preprocessing(dataset1, True)
    data3, label3 = preprocessing(dataset3, False)
    np.savetxt('data1.txt', data1)
    np.savetxt('data3.txt', data3)
    np.savetxt('label1.txt', label1)
    np.savetxt('label3.txt', label3)
    '''
    data1 = np.loadtxt('data1.txt')
    data3 = np.loadtxt('data3.txt')
    label1 = np.loadtxt('label1.txt')
    label3 = np.loadtxt('label3.txt')
    
    for i in range(data1.shape[0]):
        max_value = data1[i, :].max()
        if max_value >= 1:
            data1[i, :] /= max_value
    for i in range(data3.shape[0]):
        max_value = data3[i, :].max()
        if max_value >= 1:
            data3[i, :] /= max_value

    data = np.concatenate((data1, data3))
    label = np.concatenate((label1, label3))
    tmp = np.concatenate((data, label), axis=1)
    np.random.shuffle(tmp)
    data = tmp[:, 0:-3]
    label = tmp[:, -3:]
    np.savetxt('data.txt', data)
    np.savetxt('label,txt', label)
    np.savetxt('data_left.txt', data1)
    np.savetxt('label_left.txt', label1[:, 2])
    '''
    data4 = preprocessing(dataset4, False)
    for i in range(data4.shape[0]):
        max_value = data4[i, :].max()
        if max_value >= 1:
            data4[i, :] /= max_value
    np.savetxt('data4_normalized.txt', data4)
    '''
