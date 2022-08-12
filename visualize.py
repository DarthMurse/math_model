# This script aims to visualize the behaviors of users. Drawing graphs from csv files in the data folder.

'''
About the csv files:
    The record has an interval of 3 days.
    There are 3 main categories of resources: computation, storage and network.
    Each main category contains some metrics represented by numbers. For examples:
        computation: 3, 8, 10, 13
        storage: 1, 2, 5, 11, 12, 15
        network: 4, 6, 7, 14
    Each attribute has a number of form X-Y-Z (1-2-3 for example)
    X is the main metric number, Y-Z is the sub metric number in the main metric X
'''

def main_metric_seq(csv_num, user_num, metric_num):
    '''
    This function will plot all the sub metrics in the main metric of a user appeared in a csv file following time sequences.
    And it will also mark the point that the user is about to stop using the service(data in 2.csv)
    Parameters: 
        csv_num: the csv file number that contains the user (e.g. 1-'1.csv')
        user_num: the user to be tracked (e.g. 9-'user 9')
        metric_num: the main metric of the user to be plotted following time sequence
    '''
    # code start here
