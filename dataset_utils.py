import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import pickle
import pandas as pd
from scipy.stats import dirichlet, multinomial
#from sklearn.preprocessing import Normalizer

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ManageDatasets():

	def __init__(self, cid, dataset_name):
		self.cid = cid
		self.dataset_name = dataset_name

	def load_from_keras(self,
		dataset_size, #for each client
		alpha, #param of dirichlet distribuition
		test_size = 0.2): # proportion of dataset to use as test set

		if self.dataset_name == 'EMNIST':
			(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
													'emnist/balanced',
													split=['train', 'test'],
													batch_size=-1,
													as_supervised=True,
												))
		else:
			dataset = self.get_dataset_from_keras()      
			(x_train, y_train), (x_test, y_test) = dataset.load_data()	

		n_classes = len(np.unique(y_train)) #number of classes in dataset

		alpha_vector = alpha * np.ones(n_classes)

		client_proportions = dirichlet.rvs(alpha_vector, size = 1)
		client_quantities = []

		client_quantities = multinomial.rvs(n = dataset_size, p = client_proportions[0]) #quantity of each class

		index_train = []
		index_test = []
		for i, n in enumerate(client_quantities): #pass for all classes

			try:
				index_train = np.append(index_train, 
									np.random.choice(np.where(y_train == i)[0], int(n * (1 - test_size)))) #choose the exact quantity of each class, randomly
			except ValueError: #the client may not have a label
				pass

			try:
				index_test = np.append(index_test, 
									np.random.choice(np.where(y_test == i)[0], int(n * test_size)))
			except ValueError:
				pass
		
		index_train = index_train.astype(int)
		index_test = index_test.astype(int)

		x_train = x_train[index_train]
		y_train = y_train[index_train]

		x_test = x_test[index_test]
		y_test = y_test[index_test]

		if self.dataset_name in ['MNIST','CIFAR10', 'CIFAR100', 'FMNIST']:
			x_train = x_train/255
			x_test = x_test/255

		return x_train, y_train, x_test, y_test


	def load_UCIHAR(self):
		with open(f'data/UCI-HAR/{self.cid +1}_train.pickle', 'rb') as train_file:
			train = pickle.load(train_file)

		with open(f'data/UCI-HAR/{self.cid+1}_test.pickle', 'rb') as test_file:
			test = pickle.load(test_file)

		train['label'] = train['label'].apply(lambda x: x -1)
		y_train        = train['label'].values
		train.drop('label', axis=1, inplace=True)
		x_train = train.values

		test['label'] = test['label'].apply(lambda x: x -1)
		y_test        = test['label'].values
		test.drop('label', axis=1, inplace=True)
		x_test = test.values

		return x_train, y_train, x_test, y_test
	
	def load_ExtraSensory(self):
		with open(f'data/ExtraSensory/x_train_client_{self.cid+1}.pickle', 'rb') as x_train_file:
			x_train = pickle.load(x_train_file)

		with open(f'data/ExtraSensory/x_test_client_{self.cid+1}.pickle', 'rb') as x_test_file:
			x_test = pickle.load(x_test_file)
	    
		with open(f'data/ExtraSensory/y_train_client_{self.cid+1}.pickle', 'rb') as y_train_file:
			y_train = pickle.load(y_train_file)

		with open(f'data/ExtraSensory/y_test_client_{self.cid+1}.pickle', 'rb') as y_test_file:
			y_test = pickle.load(y_test_file)

		y_train = np.array(y_train) + 1
		#print('------------------------------', len(y_train), np.max(y_train))
		y_test  = np.array(y_test) + 1

		return x_train, y_train, x_test, y_test


	def load_MotionSense(self):
		with open(f'data/motion_sense/{self.cid+1}_train.pickle', 'rb') as train_file:
			train = pd.read_pickle(train_file)
	    
		with open(f'data/motion_sense/{self.cid+1}_test.pickle', 'rb') as test_file:
			test =  pd.read_pickle(test_file)
	        
		y_train = train['activity'].values
		train.drop('activity', axis=1, inplace=True)
		train.drop('subject', axis=1, inplace=True)
		train.drop('trial', axis=1, inplace=True)
		x_train = train.values

		y_test = test['activity'].values
		test.drop('activity', axis=1, inplace=True)
		test.drop('subject', axis=1, inplace=True)
		test.drop('trial', axis=1, inplace=True)
		x_test = test.values
	    
		return x_train, y_train, x_test, y_test
	
	def get_dataset_from_keras(self):
		#Loads a dataset from Keras based on its name and returns the class of dataset

		if self.dataset_name == 'MNIST':
			return tf.keras.datasets.mnist

		elif self.dataset_name == 'CIFAR10':
			return tf.keras.datasets.cifar10

		elif self.dataset_name == 'CIFAR100':
			return tf.keras.datasets.cifar100

		elif self.dataset_name == 'FMNIST':
			return tf.keras.datasets.fashion_mnist

		elif self.dataset_name == 'IMDB':
			return tf.keras.datasets.imdb

		elif self.dataset_name == 'REUTERS':
			return tf.keras.datasets.reuters

	def select_dataset(self,
					dataset_size = None,
					alpha = None, 
					test_size = 0.2):

		if self.dataset_name == 'MotionSense':
			return self.load_MotionSense()	
			
		elif self.dataset_name == 'ExtraSensory':
			return self.load_ExtraSensory()
		
		elif self.dataset_name == 'UCIHAR':
			return self.load_UCIHAR()
			
		elif self.dataset_name in ['MNIST','CIFAR10', 'CIFAR100',
							 	   'FMNIST', 'IMDB', 'REUTERS',
								   'EMNIST']:
			return self.load_from_keras(dataset_size = dataset_size, alpha = alpha, test_size=test_size)
