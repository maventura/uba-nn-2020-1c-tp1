import sys
import os
import numpy as np
import pickle
import pandas as pd



arguments = len(sys.argv) - 1
if arguments < 2:
	print ("Se necesitan 2 argumentos: nombre de archivo con modelo entrenado o vacio, y path de archivo csv de datos" )
	sys.exit()

pkl_file = sys.argv[1] # file for storing or recovering trained network
csv_file = sys.argv[2] # csv data file

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
df = pd.read_csv(csv_file, sep=',',header=None).to_numpy()

data = df[:,:8]
z = df[:,8:]

n = len(data[0])
m = len(z[0])
p = len(data)

x_full = np.zeros((p,n+1))
x_full[:,:-1] = data
x_full[:,-1] = 1

#x[:,1] = np.sqrt(x[:,1])
#x[:,3] = np.sqrt(x[:,3])
#x[:,6] = np.sqrt(x[:,6])

x = x_full / x_full.max(axis=0)
w = np.random.normal( 0, 0.1, (n+1,m))

lr_a = 0.0001
lr_b = 0.0005

decay = 0.001

momentum = True
E = []

e = 1
t = 0
epochs = 25000
out_modulus = epochs/8
lr = lr_a

y = np.dot( x, w)
d = z-y
dw = lr*np.dot( x.T, d)

try:
	#restoring trained network
	infile = open(pkl_file,'rb')
	w = pickle.load(infile)
	infile.close()
	#testing
	acum = 0
	for i in range(x.shape[0]):
		out = np.dot(x[i],w)
		dif = np.mean( np.square(out - z[i]))
		acum = acum + dif
	print("ECM promedio: " + str(acum/x.shape[0]) )
except:

	#TRAINING
	while (e>0.01) and (t<epochs):
		y = np.dot( x, w)
		d = z-y
		dw_old = dw
		dw_new = lr*np.dot( x.T, d)
		dw = dw_new + (1-decay)*dw_old
		if not momentum:
			dw = dw_new
		if t == epochs/2:
			lr = lr_b
		w += dw
		e = np.mean( np.square( d))
		E.append( e)
		t += 1
		if t%out_modulus==0:
			print( t, e)

	#save trained model
	outfile = open(pkl_file,'xb')
	pickle.dump(w, outfile)
	outfile.close()