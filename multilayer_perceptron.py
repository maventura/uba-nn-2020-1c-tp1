import numpy as np
import os

COL_AXIS = 1

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def step(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > 0:
                x[i][j] = 1
            else:
                x[i][j] = 0
    return x

################################################

class Perceptron():

    def __init__(self, layers, activation = "tanh"):
        self.activation = activation
        self.layers = layers
        self.l = len(layers)
        a, b = 0.0, 0.5
        self.w = [np.random.normal(a,b,(layers[i]+1,layers[i+1])) for i in range(0,len(layers)-1)] 
        self.dw_old = [np.random.normal(a,b,(layers[i]+1,layers[i+1])) for i in range(0,len(layers)-1)]
        self.decay = 0.3
        self.w.insert(0, np.empty(shape=(0, 0)))
        self.dw_old.insert(0, np.empty(shape=(0, 0)))
        self.errors = []

        
    def add_bias(self, xh):
        return np.concatenate( ( xh, np.ones((xh.shape[0],1)) ) ,axis=COL_AXIS) #adding col for threshold 
        
      
    def subBias(self, v):
        return np.delete(v, v.shape[1]-1, axis=COL_AXIS)

    
    def activation_function(self, yb):
        shape = yb.shape
        yb = np.asarray([float(a) for a in yb[0]]).reshape(shape)
        if self.activation == "sigmoid":
            yb = sigmoid(yb)
        elif self.activation == "step":
            yb = step(yb)
        elif self.activation == "tanh":
            yb = np.tanh(yb)
        else:
            print("ERROR: invalid activation method.")
        return yb
        
        
    def activate(self, xh):    #xh es un vector de dimension (#cols,)
        y  = [np.zeros(self.layers[i]).reshape((1,self.layers[i])) for i in range(0,len(self.layers))] # y debe tener tamaño #layers
        yb = xh.reshape((1,len(xh)))
        
        for k in range(1,self.l):
            y[k-1] = self.add_bias(yb)
            yb = np.dot(y[k-1],self.w[k])
            yb = self.activation_function(yb)
            
        y[-1] = np.array(yb)
        return y


    def correction(self,nu,y,zh):
        e = zh.reshape((1,len(zh))) - y[-1]
        dw = [np.zeros((self.layers[i]+1,self.layers[i+1])) for i in range(0,len(self.layers)-1)]
        dw.insert(0,[])          #dW con tamaño #layers
        d =  [0 for i in range(self.l)]
        ones = np.ones(y[-1].shape)
        dy = ones - np.square(y[-1]) #debe ser producto posicion a posicion
        d[self.l-1] = np.multiply(e,dy)

        for k in range(self.l-1, 0, -1): 
            dw[k] = nu * np.matmul(y[k-1].T, d[k])
            e = np.matmul(d[k],self.w[k].T)
            dy = np.ones(y[k-1].shape) - np.square(y[k-1])
            d[k-1] = self.subBias( np.multiply(e,dy) )
        #dw_old_decayed = [(1-self.decay)*k for k in self.dw_old]
        #res = [a + b for a, b in zip(dw, dw_old_decayed)]
        #self.dw_old = res
        #return res
        return dw
        
        
    def predict(self, x):
        return self.activate(x)[-1]
    
    
    def train(self, x, z, epochs=4000, nu=0.001, epsilon = 0.01, decay = 0.1):
        self.decay = decay
        p = x.shape[0]
        t=1
        out_modulus=epochs/8
        err = []
        e=1.0
        print("epoch - error")
        while t < epochs and e > epsilon:
            e = 0
            for h in range(p):
                y = self.activate(x[h])
                dw = self.correction(nu,y,z[h])
                self.w = [a + b for a, b in zip(self.w, dw)]
                e += np.linalg.norm(y[-1]-z[h].reshape((1,len(z[h]))))
            err.append(e)
            t+=1
            if t%out_modulus==0:
                print( t, e)
        return err