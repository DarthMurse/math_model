from preprocessing import *
import matplotlib.pyplot as plt

def draw_single_metric(user_name, dataset, title, metric, is_dataset1=True):
    pd_series = extract_single_seq(user_name, metric, dataset)
    date = pd_series.index
    values = pd_series.values
    drop_date = user_drop_date(user_name)
    plt.plot(date, values, label=metric)

    if is_dataset1:
        for day in drop_date:
            day = pd.to_datetime(day)
            plt.vlines(day, 0, max(values), linestyle='dashed')

    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()

def draw_multiple_metric(user_name, dataset, title, metrics, is_dataset1=True):
    for i in metrics:
        draw_single_metric(user_name, dataset, title, i, is_dataset1)

def draw_user_data(user_name, dataset, title, rsc_type, is_dataset1=True):
    if rsc_type == 'computation':
        draw_multiple_metric(user_name, dataset, title, [3, 8, 10, 13], is_dataset1)
    elif rsc_type == 'storage':
        draw_multiple_metric(user_name, dataset, title, [1, 2, 5, 11, 12, 15], is_dataset1)
    elif rsc_type == 'network':
        draw_multiple_metric(user_name, dataset, title, [4, 6, 7, 14], is_dataset1)
    else:
        print('not a proper type of resource')

#draw_multiple_metric('User 9', dataset1, 'User 9 computation', [3, 8, 10, 13])
#draw_single_metric('User 9', dataset1, 'User 9 metric 3', 13)
draw_user_data('User 28', dataset1, 'User 28', 'storage')
plt.show()
