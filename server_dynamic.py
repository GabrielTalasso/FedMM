import flwr as fl
import tensorflow as tf
import tensorflow_datasets as tfds
from logging import WARNING
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from keras.layers import Input, Dense, Activation
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
from server_utils import *

from dataset_utils import ManageDatasets
from model_definition import ModelCreation

from sys import getsizeof

class FedMM_dynamic(fl.server.strategy.FedAvg):

  def __init__(self, model_name, n_clients,
                clustering, clustering_rounds, dataset, 
                fraction_fit, selection_method, cluster_metric,
                cluster_method,
                metric_layer = -1,
                n_clusters = 1,
                POC_perc_of_clients = 0.5,
                decay_factor = 0.1,
                dataset_n_classes = 10,
                dir_alpha = 100,
                server_dataset_size = 1000,
                server_dataset_type = 'data',
                simulation_alias = 'simulation'):

    self.model_name = model_name
    self.n_clusters = n_clusters
    self.n_clients = n_clients
    self.clustering = clustering
    self.clustering_rounds = clustering_rounds
    self.dataset = dataset
    self.dataset_n_classes = dataset_n_classes
    self.dir_alpha = dir_alpha

    self.selection_method = selection_method
    self.POC_perc_of_clients = POC_perc_of_clients
    self.cluster_metric = cluster_metric
    self.metric_layer = metric_layer
    self.cluster_method = cluster_method
    self.simulation_alias = simulation_alias  

    self.decay_factor = decay_factor

    self.acc = []
    self.times_selected = list(np.zeros(n_clients))

    self.idx = list(np.zeros(n_clients))

    self.x_servidor = load_server_data(dataset_name = self.dataset,
                                      server_dataset_size = server_dataset_size,
                                      server_dataset_type = 'data')

    super().__init__(fraction_fit=1, 
		    min_available_clients=self.n_clients, 
		    min_fit_clients=self.n_clients, 
		    min_evaluate_clients=self.n_clients)
    
    
  def aggregate_fit(self, server_round, results, failures):

    def create_model(self):
      input_shape = self.x_servidor.shape

      if self.model_name == 'DNN':
        return ModelCreation().create_DNN(input_shape, self.dataset_n_classes)

      elif self.model_name == 'CNN':
        return ModelCreation().create_CNN(input_shape, self.dataset_n_classes)

    modelo = create_model(self)

    """Aggregate fit results using weighted average."""
    lista_modelos = {'cids': [], 'models' : {}}

    weights_results = {}
    lista_last = []
    lista_last_layer = []

    for _, fit_res in results:

      client_id = str(fit_res.metrics['cliente_id'])
      parametros_client = fit_res.parameters
      lista_modelos['cids'].append(client_id)
      idx_cluster = self.idx[int(client_id)]

      #save model weights in clusters (or create a new cluster)
      if str(idx_cluster) not in lista_modelos['models'].keys(): 
        lista_modelos['models'][str(idx_cluster)] = []
      lista_modelos['models'][str(idx_cluster)].append(parameters_to_ndarrays(parametros_client))
      
      #save model weights and the numer of examples in each client (to avg) in clusters 
      if str(idx_cluster) not in weights_results.keys():
          weights_results[str(idx_cluster)] = []
      weights_results[str(idx_cluster)].append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

      #collect activations and weights of last-layer of clients's model (or any other layer defined in metric_layer)
      w = lista_modelos['models'][str(idx_cluster)][-1]#
      modelo.set_weights(w)#
      activation_last = get_layer_outputs(modelo, modelo.layers[self.metric_layer], self.x_servidor, 0)#
      lista_last.append(activation_last)#
      lista_last_layer.append(modelo.layers[self.metric_layer].weights[0].numpy().flatten())#

    lista_modelos['actv_last'] = lista_last.copy()
    lista_modelos['last_layer'] = lista_last_layer

    #similarity between clients (construct the similatity matrix)
    if (server_round == 1) or (server_round in self.clustering_rounds):
        matrix = calcule_similarity(models = lista_modelos,
                                    metric = self.cluster_metric)
        
    #use some clustering method in similarity metrix
    if self.clustering:
      if server_round in self.clustering_rounds:
        self.idx = make_clusters(matrix = matrix,
                                clustering_method = self.cluster_method,
                                models = lista_last_layer,
                                plot_dendrogram=True,
                                n_clients = self.n_clients,
                                n_clusters=self.n_clusters, 
                                server_round = server_round,
                                clustering_rounds = self.clustering_rounds,
                                path = f'local_logs/{self.dataset}/alpha_{self.dir_alpha}/{self.cluster_metric}-({self.metric_layer})-{self.cluster_method}-{self.selection_method}-{self.POC_perc_of_clients}/')

        filename = f"local_logs/{self.dataset}/alpha_{self.dir_alpha}/{self.simulation_alias}-{self.cluster_metric}-({self.metric_layer})-{self.cluster_method}-{self.selection_method}-{self.POC_perc_of_clients}/clusters_{self.n_clients}clients_{self.n_clusters}clusters.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a') as arq:
          for i in range(len(self.idx)):
            arq.write(f"{i}, {self.idx[i]}, {server_round}\n")
        
    #aggregation params for each cluster
    parameters_aggregated = {}
    for idx_cluster in weights_results.keys():   
      parameters_aggregated[idx_cluster] = ndarrays_to_parameters(aggregate(weights_results[idx_cluster]))

    metrics_aggregated = {}
    return parameters_aggregated, metrics_aggregated

  def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        self.acc = list(range(self.n_clients))
        for c, evaluate_res in results:
           self.acc[int(c.cid)] = evaluate_res.metrics['accuracy']

        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
  
  def configure_fit(
        self, server_round, parameters, client_manager):

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        #select clients 
        clients = sample(clients = client_manager.clients,
            num_clients=sample_size, min_num_clients=min_num_clients,
            selection = self.selection_method,
            idx = self.idx, 
            server_round = server_round,
            clustering_rounds = self.clustering_rounds,
            POC_perc_of_clients = self.POC_perc_of_clients,
            acc = self.acc,
            times_selected=self.times_selected,
            decay_factor=self.decay_factor
        )
        
        for c in clients:
           self.times_selected[int(c.cid)] += 1

        config = {"round" : server_round}

        if server_round == 1:
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        
        elif server_round <= np.min(self.clustering_rounds)+1:
            fit_ins = FitIns(parameters['0.0'], config)
            return [(client, fit_ins) for client in clients]   

        else:
          return [(client, FitIns(parameters[str(self.idx[int(client.cid)])], config)) for client in clients]
  
  def configure_evaluate(
      self, server_round, parameters, client_manager):

      if self.fraction_evaluate == 0.0:
          return []
      
      config = {'round': server_round} 

      if self.on_evaluate_config_fn is not None:
          config = self.on_evaluate_config_fn(server_round)

      sample_size, min_num_clients = self.num_evaluation_clients(
          client_manager.num_available()
      )
      clients = client_manager.sample(
          num_clients=sample_size, min_num_clients=min_num_clients, 
      )
      if server_round == 1:
        evaluate_ins = EvaluateIns(parameters['0.0'], config)
        return [(client, evaluate_ins) for client in clients]
      
      elif server_round <= np.min(self.clustering_rounds):
        evaluate_ins = EvaluateIns(parameters['0.0'], config)
        return [(client, evaluate_ins) for client in clients]  

      else:
        return [(client, EvaluateIns(parameters[str(self.idx[int(client.cid)])], config)) for client in clients]