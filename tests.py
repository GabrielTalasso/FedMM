import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial as spc
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

data = pd.read_csv(f'/home/gabriel.talasso/FedMM/local_logs/CIFAR10/alpha_0.1/CKA-(-1)-HC-All-0.5/evaluate/acc_10clients_3clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data.groupby('round').mean(), y = 'acc', x = 'round', label = 'HC-10e')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()

plt.savefig('figures/teste.png')