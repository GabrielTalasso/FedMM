import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D, Dense, InputLayer, BatchNormalization, Dropout

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#from sklearn.linear_model import LogisticRegression
import numpy as np

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ModelCreation():

	def create_DNN(self, input_shape, num_classes):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(input_shape[1:])))
		model.add(tf.keras.layers.Dense(128, activation='relu'))
		model.add(tf.keras.layers.Dense(64,  activation='relu'))
		model.add(tf.keras.layers.Dense(32,  activation='relu'))
		model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return model


	def create_CNN(self, input_shape, num_classes):
		
		deep_cnn = Sequential()
		deep_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu',kernel_initializer='he_uniform', input_shape=(input_shape[1], 1)))
		deep_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu',kernel_initializer='he_uniform'))
		deep_cnn.add(Dropout(0.6))
		deep_cnn.add(MaxPooling1D(pool_size=2))
		deep_cnn.add(Flatten())
		deep_cnn.add(Dense(50, activation='relu'))
		deep_cnn.add(Dense(num_classes, activation='softmax'))
	
		deep_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return deep_cnn





