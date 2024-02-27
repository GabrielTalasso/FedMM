import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

dataset = 'CIFAR10'
alpha = '0.1'

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_20clients_4clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-Random-All-0.5/evaluate/acc_20clients_4clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

# data = pd.read_csv(f'local_logs/{dataset}/alpha_{alpha}/CKA-(-1)-HC-All-0.5/evaluate/acc_30clients_1clusters.csv',
#                     names = ['round', 'cid', 'acc', 'loss'])

# plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round')
# plt.ylabel("Acuracy")
# plt.xlabel("Rounds")
# plt.show()
plt.savefig('figures/comparison.png')
    