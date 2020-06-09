import sys
import os
import numpy as np
import pickle
import pandas as pd
from multilayer_perceptron import Perceptron


def roundToInt(x):
	if(x > 0.5):
		return 1
	else:
		return 0

arguments = len(sys.argv) - 1
if arguments < 2:
	print ("Se necesitan 2 argumentos: nombre de archivo con modelo entrenado o vacio, y path de archivo csv de datos" )
	sys.exit()


pkl_file = sys.argv[1] # file for storing or recovering trained network
csv_file = sys.argv[2] # csv data file

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
arr = pd.read_csv(csv_file, sep=',',header=None).to_numpy()
x_full = arr[:,1:]
x = x_full / x_full.max(axis=0) #normalize data
z = np.asarray([ [1] if zi == 'M' else [0] for zi in arr[:,:1] ])



try:
	#restoring trained network
	infile = open(pkl_file,'rb')
	trained_net = pickle.load(infile)
	infile.close()
	#take input to train
	acum = 0
	for i in range(0,x.shape[0]):
		xi = roundToInt(trained_net.predict(x[i] ) )
		if (xi == z[i]):
			acum = acum + 1

	print("Accuracy: " + str( (acum/x.shape[0]) * 100) )

except Exception as e:     
	layers = np.array([10,20,20,20,20,20,1])
	network = Perceptron(layers)
	err = network.train(x, z, epochs=8000, nu=0.05)
	#saving trained network
	outfile = open(pkl_file,'xb')
	pickle.dump(network, outfile)
	outfile.close()


