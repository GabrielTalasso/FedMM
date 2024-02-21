import flwr as fl
import pickle
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

import tensorflow as tf
import numpy as np
import random
import os

from dataset_utils import ManageDatasets
from model_definition import ModelCreation

from sys import getsizeof

class ClientBase(fl.client.NumPyClient):

	def __init__(self, cid, dataset, n_clients, model_name, local_epochs,
	      n_rounds, n_clusters, selection_method, cluster_metric, 
		  cluster_method, metric_layer = -1, 
		  POC_perc_of_clients = 0.5,
		  dir_alpha = 100,
		  dataset_size = 1000,
		  dataset_n_classes = 10):

		self.cid = cid
		self.round = 0
		self.n_clients = n_clients

		self.dataset = dataset
		self.dir_alpha = dir_alpha #dirichlet alpha
		self.dataset_size = dataset_size
		self.dataset_n_classes = dataset_n_classes

		self.model_name = model_name
		self.local_epochs = local_epochs

		self.selection_method = selection_method
		self.POC_perc_of_clients = POC_perc_of_clients
		self.cluster_metric = cluster_metric
		self.metric_layer = metric_layer
		self.cluster_method = cluster_method

		self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
		self.model     = self.create_model()

		self.n_rounds = n_rounds
		self.n_clusters = n_clusters

	def load_data(self):
		return ManageDatasets(self.cid, dataset_name=self.dataset).select_dataset(self.n_clients,
																				  alpha = self.dir_alpha,
																				  dataset_size = self.dataset_size)

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape, self.dataset_n_classes)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, self.dataset_n_classes)
		

	def get_parameters(self, config):
		return self.model.get_weights()

	def fit(self, parameters, config):

		self.model.set_weights(parameters)
		h = self.model.fit(self.x_train, self.y_train, 
		                    validation_data = (self.x_test, self.y_test),
							verbose=1, epochs=self.local_epochs)

		msg2server = {
				"cliente_id": self.cid,
		}
		self.round += 1
	
		filename = f"local_logs/{self.dataset}/{self.cluster_metric}-({self.metric_layer})-{self.cluster_method}-{self.selection_method}-{self.POC_perc_of_clients}/train/acc_{self.n_clients}clients_{self.n_clusters}clusters.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{config['round']}, {self.cid}, {np.mean(h.history['accuracy'])}, {np.mean(h.history['loss'])}\n")

		return self.model.get_weights(), len(self.x_train), msg2server


	def evaluate(self, parameters, config):
		self.model.set_weights(parameters)

		loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
		filename = f"local_logs/{self.dataset}/{self.cluster_metric}-({self.metric_layer})-{self.cluster_method}-{self.selection_method}-{self.POC_perc_of_clients}/evaluate/acc_{self.n_clients}clients_{self.n_clusters}clusters.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{config['round']}, {self.cid}, {accuracy}, {loss}\n")
	
	
		return loss, len(self.x_test), {"accuracy" : accuracy}