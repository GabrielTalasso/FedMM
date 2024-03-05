import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

def plot_accuracy_separate(datasets, n_clients, n_clusters, alpha, methods = ['HC', 'Random', 'Fedavg']):

    fig, ax = plt.subplots(1,3, figsize=(10, 5))
  

    for i, method in enumerate(methods):

        label = method

        if method == 'HC':
            label = 'FedMM'

        if method == 'Fedavg':
            n_clusters = 1
            method = 'Random'
            label = 'Fedavg'
        
        data = pd.read_csv(f'local_logs/{datasets}/alpha_{alpha}/CKA-(-1)-{method}-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

    
        plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', ax = ax[i])# , label = label)
        ax[i].set_title(label)
        ax[i].set_yticks((0,1))

    plt.ylabel("Acuracy")
    plt.xlabel("Rounds")
    plt.show()
    plt.savefig('figures/comparison.png')


def plot_accuracy_together(datasets, n_clients, n_clusters, alpha, methods = ['HC', 'Random', 'Fedavg'], extra_name = None, extra_path = None):

    for i, method in enumerate(methods):

        label = method

        if method == 'HC':
            label = 'FedMM'

        if method == 'Fedavg':
            n_clusters = 1
            method = 'Random'
            label = 'Fedavg'


        if method == 'Extra':
            n_clusters = 10
            data = pd.read_csv(f'{extra_path}/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])
            
            label = extra_name

        else:
        
            data = pd.read_csv(f'local_logs/{datasets}/alpha_{alpha}/CKA-(-1)-{method}-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                        names = ['round', 'cid', 'acc', 'loss'])

    
        plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = label)

    plt.ylabel("Acuracy")
    plt.xlabel("Rounds")
    plt.show()
    plt.savefig('figures/comparison_together.png')


dataset = 'CIFAR10'
alpha = '0.1'
n_clients = 50
n_clusters = 10

plot_accuracy_together(dataset, n_clients, n_clusters, alpha, 
                        methods = ['HC', 'Random', 'Fedavg', 'Extra'],
                        extra_name = 'w_data', extra_path = 'local_logs/CIFAR10/alpha_0.1/CKA-(-1)-HC-All-0.55')
plot_accuracy_separate(dataset, n_clients, n_clusters, alpha)


# #for diferent number of clusters
# for c in [1,2,5,10,15,20,25]:
#     data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_{n_clients}clients_{c}clusters.csv',
#                         names = ['round', 'cid', 'acc', 'loss'])
#     sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = c)
#     plt.ylim(0,1)
#     plt.ylabel("Acuracy")
#     plt.xlabel("Rounds")
#     plt.savefig('figures/comparison_nclusters.png')


# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.555/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC_5e')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-1e')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5555/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-10e')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# plt.savefig('figures/comparison_epochs.png')
    