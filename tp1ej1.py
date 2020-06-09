import sys
import os
import numpy as np
import pickle
import pandas as pd
from multilayer_perceptron import Perceptron


#imprimir ayuda de uso, tipo 

#obtener parametros de entrada (es entrenamiento o testeo,1) 
# python tp1ej1.py nuevo_modelo tp1_ej1_training.csv

# Si el archivo "nuevo_modelo" no existe, entonces el programa debería
# usar los datos en "tp1_ej1_training.csv" para entrenar un modelo nuevo,
# mostrando alguna información de su desempeño durante el entrenamiento, y
# guardar el modelo entrenado en el nuevo archivo.

# 2) python tp1ej1.py modelo_entrenado tp1_ej1_testing.csv)

# Si el archivo "modelo_entrenado" existe, entonces debería cargar este
# modelo y usar los datos en "tp1_ej1_testing.csv" para hacer un testeo,
# mostrando información sobre el desempeño del modelo.
# Count the arguments


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
	y = trained_net.predict(x)
	print(y)
except Exception as e:     
	#train new network
	layers = np.array([10,5,3,1])
	network = Perceptron(layers)
	err = network.train(x, z, epochs=8000, nu=0.05)
	#saving trained network
	outfile = open(pkl_file,'xb')
	pickle.dump(network, outfile)
	outfile.close()


