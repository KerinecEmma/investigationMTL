import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential

#Models for learning

def multi_tasks_multi_layer_perceptron(NB_FEATURES, size_layers):

	print('Building model multi-tasks...')

	main_input = Input(shape = (NB_FEATURES,))
	aux_input = Input(shape = (NB_FEATURES,))

	shared_layer1 = Dense(size_layers[0], input_dim = NB_FEATURES)
	main_h1 = shared_layer1(main_input)
	main_h2 = Activation('sigmoid')(main_h1)
	main_h3 = Dropout(0.5)(main_h2)
	aux_h1 = shared_layer1(aux_input)
	aux_h2 = Activation('sigmoid')(aux_h1)
	aux_h3 = Dropout(0.5)(aux_h2)

	shared_layer2 = Dense(size_layers[0], input_dim = NB_FEATURES)
	main_h4 = shared_layer2(main_h3)
	main_h5 = Activation('sigmoid')(main_h4)
	main_h6 = Dropout(0.5)(main_h5)
	aux_h4 = shared_layer2(aux_h3)
	aux_h5 = Activation('sigmoid')(aux_h4)
	aux_h6 = Dropout(0.5)(aux_h5)

	final1 = Dense(1, init = "uniform")
	final2 = Dense(1, init = "uniform")
	main_7 = final1(main_h6)
	main_output = Activation('sigmoid')(main_7)
	aux_7 = final2(aux_h6)
	aux_output = Activation('sigmoid')(aux_7)

	model = Model(inputs = [main_input,aux_input], outputs = [main_output, aux_output])

	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return(model)


def simple_multi_layer_perceptron(NB_FEATURES, size_layers):

	print('Building model simple...')
	model = Sequential()
	
	model.add(Dense(size_layers[0], input_dim = NB_FEATURES))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))

	model.add(Dense(size_layers[1], input_dim = NB_FEATURES))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))

	model.add(Dense(1, init = "uniform"))
	model.add(Activation('sigmoid'))

	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return(model)
