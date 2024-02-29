import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

dataset = 'CIFAR10'
alpha = '0.1'
n_clients = 30
n_clusters = 10

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC1')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-2)-HC-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC2')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.55/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-noise1')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-Random-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'rand1')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-2)-Random-All-0.5/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'rand2')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-2)-HC-All-0.55/evaluate/acc_{n_clients}clients_{n_clusters}clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-noise2')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()


data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_{n_clients}clients_1clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'fedavg')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()
plt.savefig('figures/comparison.png')
    