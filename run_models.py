import numpy as np
import math
import keras
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
from keras.models import Model
from utils import *
from models import *



#set the parameters of the experiment:

nb_features= 10000 #for tf-idf
nb_ex = 700 #for testing
nb_train = 200 #for training
size_layers = [100, 100] #of models
nb_epochs = 150 #for training
number_test = 0
batch = 32
exp = 1

tasks, list_ind, list_exp = creation(exp)
print(list_exp)

class MetricsSimple(keras.callbacks.Callback):
	def on_train_begin(self, logs = {}):
		self.losses = []
		return()

	def on_epoch_end(self, epoch, logs = {}):
		y_pred = self.model.predict(self.validation_data[0])
		y_true = self.validation_data[1]
		y_predi = turn_int(y_pred)
		self.acc = accuracy_score(y_true, y_predi)
		self.f1 = f1_score(y_true, y_predi, average='micro')
		self.prec = precision_score(y_true, y_predi, average='micro')
		self.rec = recall_score(y_true, y_predi, average='micro')
		self.losses.append(logs.get('loss'))
		if batch  > 1 and (epoch % 1 == 0 or epoch == nb_epochs):
			losslog.write("Epoch {}, loss {}, val_loss {}, cacc {}, acc {}, val_acc {}, f1 {}, p {}, r {}\n".format(epoch, logs.get('loss'), logs.get('val_loss'), self.acc, logs.get('acc'), logs.get('val_acc'), self.f1, self.prec, self.rec))
		return()
metricsS = MetricsSimple()

class MetricsMulti(keras.callbacks.Callback):
	def on_train_begin(self, logs = {}):
		self.losses = []
		return()

	def on_epoch_end(self, epoch, logs = {}):
		liste=sorted(logs.keys())
		y_pred = self.model.predict([self.validation_data[0], self.validation_data[1]])
		y_true = self.validation_data[2]
		y_pred[0] = turn_int(y_pred[0])
		self.acc = accuracy_score(y_true, y_pred[0])
		self.f1 = f1_score(y_true, y_pred[0], average='micro')
		self.prec = precision_score(y_true, y_pred[0], average='micro')
		self.rec = recall_score(y_true, y_pred[0], average='micro')
		#self.losses.append(logs.get('loss'))
		if batch > 1 and (epoch % 1 == 0 or epoch == nb_epochs):
			losslog.write("Epoch {}, cacc {}, f1 {}, p {}, r {} "
				.format(epoch, self.acc, self.f1, self.prec, self.rec))
			for i in liste:
				losslog.write(str(i) + " " + str(logs.get(i)) + " ")
			losslog.write("\n")

		return()

metricsM = MetricsMulti()

#run the simple models
for i, task in enumerate(tasks):
	print(task)
	print(i)

	losslog = open(name(exp, nb_train, task, number_test, False)+"true", "w")

	trainX, trainY, testX, testY = make_inputs(exp, task, nb_features, nb_train, nb_ex)
	#trainX, trainY, testX, testY, _, _ = read_sets(exp, nb_train, task)
	
	model = simple_multi_layer_perceptron(nb_features, size_layers)
	model.fit(trainX, trainY, validation_data = [testX, testY], epochs = nb_epochs, batch_size = batch, callbacks = [metricsS])
	
	losslog.close()

#run multi models
for i, ex in enumerate(list_exp):
	print(ex)
	print(i)
	task = ex[0]
	au = ex[1]

	losslog = open(name(exp, nb_train, [task, au],  number_test, False)+"true", "w")

	MtrainX, MtrainY, MtestX, MtestY, _, _ = read_sets(exp, nb_train, task)
	AtrainX, AtrainY, AtestX, AtestY, _, _ = read_sets(exp, nb_train, au)
	model = multi_tasks_multi_layer_perceptron(nb_features, size_layers)

	if len(MtestX) > len(AtestX):
		MtestX = MtestX[:len(AtestX)]
		MtestY = MtestY[:len(AtestX)]
	else:
		AtestX = AtestX[:len(MtestX)]
		AtestY = AtestY[:len(MtestX)]			

	model.fit([MtrainX, AtrainX], [MtrainY, AtrainY], validation_data = [[MtestX, AtestX], [MtestY, AtestY]], epochs = 100, batch_size = batch, callbacks = [metricsM])

	losslog.close()

