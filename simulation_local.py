from client import ClientBase
from server import FedMM
import pickle
import flwr as fl
import os
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

try:
	os.remove('./results/history_simulation.pickle')
except FileNotFoundError:
	pass

try:
	os.remove('./results/acc.csv')
except FileNotFoundError:
	pass

dataset_name = 'CIFAR10'
selection_method = 'All' #Random, POC, All, Less_Selected
cluster_metric = 'CKA' #CKA, weights
metric_layer = -1 #-1, -2, 1
cluster_method = 'Random' #Affinity, HC, KCenter, Random
POC_perc_of_clients = 0.5
n_clients = 31
n_rounds = 20
n_clusters = 6
clustering = True
cluster_round = 10
dir_alpha = 0.1
dataset_n_classes = 10
model_name = 'CNN'

def funcao_cliente(cid):
	return ClientBase(int(cid), n_clients=n_clients,
		    dataset=dataset_name, model_name = model_name,
			local_epochs = 1, n_rounds = n_rounds, n_clusters = n_clusters,
			selection_method = selection_method, 
			POC_perc_of_clients = POC_perc_of_clients,
			cluster_metric = cluster_metric,
			metric_layer = metric_layer,
			cluster_method = cluster_method,
			dir_alpha = dir_alpha,
			dataset_n_classes = dataset_n_classes)

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy=FedMM(model_name=model_name,  n_clients = n_clients, 
			     									clustering = clustering, clustering_round = cluster_round, 
													n_clusters = n_clusters, dataset=dataset_name, fraction_fit=1, 
													selection_method = selection_method, 
													POC_perc_of_clients = POC_perc_of_clients,
													cluster_metric = cluster_metric,
													metric_layer = metric_layer,
													cluster_method = cluster_method,
													dir_alpha = dir_alpha,
													dataset_n_classes = dataset_n_classes ),
								config=fl.server.ServerConfig(n_rounds))

with open('./results/history_simulation.pickle', 'wb') as file:
    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)