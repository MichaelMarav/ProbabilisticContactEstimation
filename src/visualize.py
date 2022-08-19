import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

filename = "../data/atlas_02ground.csv" 
start = 0
stop  = 5000

'''
Input:
    column: One column of data (e.g. Fz -> data[:,2])
    labels: The labels column (e.g. data[:,-1])
Output:
    normalized labels to for visualization purposes
    stable: max of the column
    fly  : (max+min)/2 
    slip : min of the column

'''
def transform_labels(column,labels):
    tr_labels = np.empty((column.shape[0],))
    max_ = np.max(column)
    min_ = np.min(column)
    mid_ = (abs(max_)+abs(min_))/2
    
    for i in range(tr_labels.shape[0]):
        if labels[i] == 0:
            tr_labels[i] = max_
        elif labels[i] == 1:
            tr_labels[i] = None
        else:
            tr_labels[i] = min_
    return tr_labels


def main():

    data = genfromtxt(filename, delimiter=',')
    time = np.arange(stop-start)
    
    labels_fz = transform_labels(data[:,2],data[:,-1])
    labels_ax = transform_labels(data[:,6],data[:,-1])
    labels_ay = transform_labels(data[:,7],data[:,-1])


    

    fig, ax = plt.subplots(3)
    ax[0].plot(time,data[start:stop,2])
    ax[1].plot(time,data[start:stop,6])
    ax[2].plot(time,data[start:stop,7])
    
    ax[0].scatter(time,labels_fz[start:stop],c = 'g')
    ax[1].scatter(time,labels_ax[start:stop],c = 'g')
    ax[2].scatter(time,labels_ay[start:stop],c = 'g')
    
    ax[0].axvline(x=1773,c='r')
    ax[1].axvline(x=1773,c='r')
    ax[2].axvline(x=1773,c='r')


    ax[0].axvline(x=2554,c='r')
    ax[1].axvline(x=2554,c='r')
    ax[2].axvline(x=2554,c='r')

    plt.show()
if __name__ == '__main__':
    main()