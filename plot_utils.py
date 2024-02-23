import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

data = pd.read_csv('local_logs/CIFAR10/alpha_0.1/CKA-(-1)-HC-All-0.5/evaluate/acc_31clients_1clusters.csv',
                    names = ['round', 'cid', 'acc', 'loss'])

plot = sns.lineplot(data, y = 'acc', x = 'round')
plt.ylabel("Acuracy")
plt.xlabel("Rounds")
plt.show()
plt.savefig('figures/test4.png')

    