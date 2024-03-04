import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as spc
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

matrix = np.array([ [1,  0.1, 0.1, 0.1],
                    [0.1, 1, 0.5, 0.9],
                    [0.1, 0.5, 1, 0.9],
                    [0.1, 0.9, 0.9, 1]])

pdist = spc.distance.pdist(matrix)
sqf = spc.distance.squareform(pdist)


linkage = linkage(pdist, method='ward')
idx = fcluster(linkage, 0.75, 'distance')

dendrogram(linkage, color_threshold=0.75)
plt.savefig('test_dendrogram.png')

print(matrix, pdist,'\n', sqf)