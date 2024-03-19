import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

import random
import threading
import math
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS

import keras.backend as K


################# for clustering

class GreedyKCenter(object):
    def fit(self, points, k):
        centers = []
        centers_index = []
        # Initialize distances
        distances = [np.inf for u in points]
        # Initialize cluster labels
        labels = [np.inf for u in points]

        for cluster in range(k):
            # Let u be the point of P such that d[u] is maximum
            u_index = distances.index(max(distances))
            u = points[u_index]
            # u is the next cluster center
            centers.append(u)
            centers_index.append(u_index)

            # Update distance to nearest center
            for i, v in enumerate(points):
                distance_to_u = self.distance(u, v)  # Calculate from v to u
                if distance_to_u < distances[i]:
                    distances[i] = distance_to_u
                    labels[i] = cluster

            # Update the bottleneck distance
            max_distance = max(distances)

        # Return centers, labels, max delta, labels
        self.centers = centers
        self.centers_index = centers_index
        self.max_distance = max_distance
        self.labels = labels

    @staticmethod
    def distance(u, v):
        displacement = u - v
        return np.sqrt(displacement.dot(displacement))

def server_Hclusters(matrix, plot_dendrogram , n_clients, n_clusters,
                    server_round, clustering_rounds, path):

    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='ward')
    min_link = linkage[0][2]
    max_link = linkage[-1][2]


    th = max_link
    for i in np.linspace(min_link,max_link, 5000):

        le = len(pd.Series(spc.fcluster(linkage, i, 'distance' )).unique())
        if le == n_clusters:
            th = i

    idx = spc.fcluster(linkage, th, 'distance')
    print(idx)

    if plot_dendrogram and (server_round in clustering_rounds):

        dendrogram(linkage, color_threshold=th)
        #plt.savefig(f'results/clusters_{dataset}_{n_clients}clients_{n_clusters}clusters.png')
        os.makedirs(os.path.dirname(path+'/dendrograms/'), exist_ok=True)
        plt.savefig(path+f'/dendrograms/clusters_{n_clients}clients_{n_clusters}clusters_round{server_round}.png')
    
    return idx

def server_AffinityClustering(matrix):

    af = AffinityPropagation(random_state=0).fit( 1 / matrix )
    idx = af.labels_

    return idx

def server_OPTICSClustering(matrix):
    clustering = OPTICS(min_samples=2).fit(1/ matrix)
    idx = clustering.labels_

def server_KCenterClustering(weights, k):

    KCenter = GreedyKCenter()
    KCenter.fit(weights, k)
    idx = KCenter.labels

    return idx


def make_clusters(matrix, plot_dendrogram , n_clients, n_clusters,
                    server_round, clustering_rounds, path, 
                    clustering_method, models):
    
    if clustering_method == 'Affinity':
        idx = server_AffinityClustering(matrix)
        return idx

    if clustering_method == 'HC':
        idx = server_Hclusters(matrix = matrix, plot_dendrogram=plot_dendrogram,
                                  n_clients=n_clients, n_clusters=n_clusters, 
                                  server_round = server_round, clustering_rounds=clustering_rounds,
                                  path = path)
        return idx
          
    if clustering_method == 'KCenter':
        idx = server_KCenterClustering(models, k = n_clusters)
        return idx

    if clustering_method == 'Random':
        unique = 0
        while unique != n_clusters:
            idx = list(np.random.randint(0, n_clusters, n_clients))
            unique = np.unique(np.array(idx))
            unique = len(unique)
        return idx


####################### for similarity

def flatten_elements(array):
    shape = array.shape
    new_shape = (shape[0], np.product(shape[1:]))
    return array.reshape(new_shape)

def get_layer_outputs(model, layer, input_data, learning_phase=1):
    layer_fn = K.function(model.input, layer.output)
    return layer_fn(input_data)

def cka(X, Y):

    # if len(X.shape) > 2:
    #     X = flatten_elements(X)
    #     Y = flatten_elements(Y)

    # Implements linear CKA as in Kornblith et al. (2019)
    X = X.copy()
    Y = Y.copy()

    # Center X and Y
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Calculate CKA
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)

    return (YTX ** 2).sum() / (np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum()))


def calcule_similarity(models, metric):
    actvs = models['actv_last']
    if metric == 'CKA':
        matrix = np.zeros((len(actvs), len(actvs)))

        for i , a in enumerate(actvs):
            for j, b in enumerate(actvs):

                x = int(models['cids'][i])
                y = int(models['cids'][j])

                matrix[x][y] = cka(a, b)

    last = models['last_layer']
    if metric == 'weights':
        matrix = np.zeros((len(last), len(last)))

        for i , a in enumerate(last):
            for j, b in enumerate(last):

                x = int(models['cids'][i])
                y = int(models['cids'][j])

                matrix[x][y] = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) #cos similarity
    return matrix


#################### for client selection
def sample(
    clients,
    num_clients: int,
    min_num_clients: Optional[int] = None,
    criterion: Optional[Criterion] = None,
    CL = True,
    selection = None,
    acc = None,
    decay_factor = None,
    server_round = None,
    idx = None,
    clustering_rounds = [0],
    POC_perc_of_clients = 0.5,
    times_selected = []):
    

    """Sample a number of Flower ClientProxy instances."""
    # Block until at least num_clients are connected.
    if min_num_clients is None:
        min_num_clients = num_clients
    # Sample clients which meet the criterion
    available_cids = list(clients)
    if criterion is not None:
        available_cids = [
            cid for cid in available_cids if criterion.select(clients[cid])
        ]

    if num_clients > len(available_cids):
        log(
            INFO,
            "Sampling failed: number of available clients"
            " (%s) is less than number of requested clients (%s).",
            len(available_cids),
            num_clients,
        )
        return []
    
    sampled_cids = available_cids.copy()
                
    
    if (idx is not None) and (server_round>np.min(clustering_rounds)) and CL:        
        selected_clients = []
        for cluster_idx in np.unique(idx): #passa por todos os clusters
            cluster = []

            for client in available_cids:
               if idx[int(client)] == cluster_idx: #salva apenas os clientes pertencentes aquele cluster
                   cluster.append(int(client))

            if selection == 'Random':
               selected_clients.append(str(random.sample(cluster,1)[0]))

            if selection == 'POC':
               acc_cluster = list(np.array(acc)[cluster]) 
               sorted_cluster = [str(x) for _,x in sorted(zip(acc_cluster,cluster))]
               clients2select = max(int(float(len(cluster)) * float(POC_perc_of_clients)), 1)
               for c in sorted_cluster[:clients2select]:
                    selected_clients.append(c)

            if selection == 'Less_Selected':
                clients_to_select_in_cluster = []
                c = [times_selected[i] for i in cluster]
                number_less_selected = min(c)
                #client_to_select = times_selected.index(number_less_selected)
                client_to_select = list(pd.Series(times_selected)[pd.Series(times_selected) == number_less_selected].index)

                for i in client_to_select: #if there are more than one with same times_selected
                    if i in cluster:
                        clients_to_select_in_cluster.append(i)

                selected_clients.append(str(random.sample(clients_to_select_in_cluster,1)[0])) #select only one per cluster

            if selection == 'DEEV' and server_round>1:
                selected_clients_cluster = []
                acc_cluster = list(np.array(acc)[cluster]) 
                for idx_accuracy in range(len(acc_cluster)):
                    if acc_cluster[idx_accuracy] <= np.mean(np.array(acc_cluster)):
                        selected_clients_cluster.append(str(cluster[idx_accuracy]))

                if decay_factor > 0:
                    the_chosen_ones  = len(selected_clients_cluster) * (1 - decay_factor)**int(server_round-np.min(clustering_rounds))
                    selected_clients_cluster = selected_clients_cluster[ : math.ceil(the_chosen_ones)]
                
                selected_clients = selected_clients + selected_clients_cluster

            # for i in client_to_select: #if there are more than one with same times_selected
            #     if i in cluster:
            #         clients_to_select_in_cluster.append(i)
            # selected_clients.append(str(random.sample(clients_to_select_in_cluster,1)[0])) #select only one per cluster
                
                
        sampled_cids = selected_clients.copy()

    if selection == 'All':
        sampled_cids = random.sample(available_cids, num_clients)  

    return [clients[cid] for cid in sampled_cids]

def load_server_data(dataset_name, server_dataset_size, server_dataset_type = 'data'):

    if server_dataset_type == 'data':

        if dataset_name == 'MNIST':
            (x_servidor, _), (_, _) = tf.keras.datasets.mnist.load_data()
            x_servidor = x_servidor/255.0
            x_servidor = x_servidor[list(np.random.random_integers(1,60000-1, server_dataset_size))]

        if dataset_name == 'EMNIST':
            (x_servidor, _), (_, _) =  tfds.as_numpy(tfds.load(
                                                            'emnist/balanced',
                                                            split=['train', 'test'],
                                                            batch_size=-1,
                                                            as_supervised=True,
                                                        ))
            x_servidor = x_servidor/255.0
            x_servidor = x_servidor[list(np.random.random_integers(1,100000-1, server_dataset_size))]

        if dataset_name == 'CIFAR10':
            (x_servidor, _), (_, _) = tf.keras.datasets.cifar10.load_data()
            x_servidor = x_servidor/255.0
            x_servidor = x_servidor[list(np.random.random_integers(1,50000-1, server_dataset_size))]
            


        if dataset_name == 'MotionSense':
            for cid in range(n_clients):
                with open(f'data/motion_sense/{cid+1}_train.pickle', 'rb') as train_file:
                    if cid == 0:
                        train = pd.read_pickle(train_file)   
                        train = train.sample(100)
                    else:
                        train = pd.concat([train,  pd.read_pickle(train_file).sample(100)],
                                            ignore_index=True, sort = False)
                    
            train.drop('activity', axis=1, inplace=True)
            train.drop('subject', axis=1, inplace=True)
            train.drop('trial', axis=1, inplace=True)
            x_servidor = train.values

        if dataset_name == 'ExtraSensory':

            for cid in range(n_clients):
                with open(f'data/ExtraSensory/x_train_client_{cid+1}.pickle', 'rb') as train_file:
                    if cid == 0:
                        train = pd.read_pickle(train_file)   
                        train = train[:20]
                    else:
                        train = np.append(train,  pd.read_pickle(train_file)[:20],
                                            axis = 0)
                    
                x_servidor = train

    if server_dataset_type == 'noise':

        if dataset_name == 'MNIST':
            (x_servidor, _), (_, _) = tf.keras.datasets.mnist.load_data()
            shape = x_servidor.shape
            x_servidor = np.random.randint(0, 1, size=shape)
            
        if dataset_name == 'EMNIST':
            (x_servidor, _), (_, _) =  tfds.as_numpy(tfds.load(
                                                            'emnist/balanced',
                                                            split=['train', 'test'],
                                                            batch_size=-1,
                                                            as_supervised=True,
                                                        ))
            shape = x_servidor.shape
            x_servidor = np.random.randint(0, 1, size=shape) 

        if dataset_name == 'CIFAR10':
            (x_servidor, _), (_, _) = tf.keras.datasets.cifar10.load_data()
            shape = x_servidor.shape
            x_servidor = np.random.randint(0, 1, size=shape) 

        if dataset_name == 'MotionSense':
            for cid in range(n_clients):
                with open(f'data/motion_sense/{cid+1}_train.pickle', 'rb') as train_file:
                    if cid == 0:
                        train = pd.read_pickle(train_file)   
                        train = train.sample(100)
                    else:
                        train = pd.concat([train,  pd.read_pickle(train_file).sample(100)],
                                            ignore_index=True, sort = False)
                    
            train.drop('activity', axis=1, inplace=True)
            train.drop('subject', axis=1, inplace=True)
            train.drop('trial', axis=1, inplace=True)
            x_servidor = train.values

        if dataset_name == 'ExtraSensory':

            for cid in range(n_clients):
                with open(f'data/ExtraSensory/x_train_client_{cid+1}.pickle', 'rb') as train_file:
                    if cid == 0:
                        train = pd.read_pickle(train_file)   
                        train = train[:20]
                    else:
                        train = np.append(train,  pd.read_pickle(train_file)[:20],
                                            axis = 0)
                    
                x_servidor = train

    return x_servidor
