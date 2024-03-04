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
            method = 'HC'
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


def plot_accuracy_together(datasets, n_clients, n_clusters, alpha, methods = ['HC', 'Random', 'Fedavg']):

    for i, method in enumerate(methods):

        label = method

        if method == 'HC':
            label = 'FedMM'

        if method == 'Fedavg':
            n_clusters = 1
            method = 'HC'
            label = 'Fedavg'
        
        data = pd.read_csv(f'local_logs/{datasets}/alpha_{alpha}/CKA-(-1)-{method}-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

    
        plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = label)

    plt.ylabel("Acuracy")
    plt.xlabel("Rounds")
    plt.show()
    plt.savefig('figures/comparison_together.png')


dataset = 'CIFAR10'
alpha = '0.1'
n_clients = 30
n_clusters = 10

plot_accuracy_together(dataset, n_clients, n_clusters, alpha)
plot_accuracy_separate(dataset, n_clients, n_clusters, alpha)


#for diferent number of clusters
for c in [1,2,5,10,15,20,25]:
    data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_{n_clients}clients_{c}clusters.csv',
                        names = ['round', 'cid', 'acc', 'loss'])
    sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = c)
    plt.ylim(0,1)
    plt.ylabel("Acuracy")
    plt.xlabel("Rounds")
    plt.savefig('figures/comparison_nclusters.png')

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-2)-HC-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC2')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.55/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-noise1')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-Random-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'rand1')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-2)-Random-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'rand2')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-2)-HC-All-0.55/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-noise2')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()


# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_{n_clients}clients_1clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'fedavg')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()
# plt.savefig('figures/comparison.png')
    