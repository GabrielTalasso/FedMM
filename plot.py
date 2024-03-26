import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial as spc
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data = pd.read_csv(f'/home/gabriel.talasso/FedMM/local_logs/CIFAR10/alpha_0.1/dynamic50-CKA-(-1)-HC-All-0.5/evaluate/acc_30clients_6clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'FedMM_dynamic')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'/home/gabriel.talasso/FedMM/local_logs/CIFAR10/alpha_0.1/static50-CKA-(-1)-HC-All-0.5/evaluate/acc_30clients_6clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'FedmMM_static')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'/home/gabriel.talasso/FedMM/local_logs/CIFAR10/alpha_0.1/fedavg-CKA-(-1)-HC-All-0.5/evaluate/acc_30clients_1clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'FedAvg')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'/home/gabriel.talasso/FedMM/local_logs/CIFAR10/alpha_0.1/random_dynamic-CKA-(-1)-Random-All-0.5/evaluate/acc_30clients_6clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'Rand_dynamic')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'/home/gabriel.talasso/FedMM/local_logs/CIFAR10/alpha_0.1/random_dynamic2-CKA-(-1)-Random-All-0.5/evaluate/acc_30clients_6clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'Rand_dynamic')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

plt.savefig('figures/teste_d.png')