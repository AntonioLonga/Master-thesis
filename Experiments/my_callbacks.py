
import keras.backend.tensorflow_backend as K
from keras import callbacks
import math
import numpy as np

class MyCallback_sinusoidal(callbacks.Callback):
    def __init__(self, w_dec, w_cla, n_epoch,plateau,k,n_period,scale_c,scale_d):
        
        self.n_epoch = n_epoch
        self.w_dec = w_dec
        self.w_cla = w_cla
        
        self.w_clas = []
        self.w_decs = []
        
        if (isinstance(scale_c,(int,float))):
            scale_c = np.ones(n_epoch)*scale_c
        
        if (isinstance(scale_d,(int,float))):
            scale_d = np.ones(n_epoch)*scale_d
            
        self.scale_c = scale_c
        self.scale_d = scale_d
        
        self.sigmoid = gen_sigm_seq(plateau,k,n_epoch,n_period)
        
        
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.w_cla, (-self.sigmoid[epoch]+1)*self.scale_c[epoch])
        K.set_value(self.w_dec, (self.sigmoid[epoch])*self.scale_d[epoch])

            
        self.w_decs.append(K.get_value(self.w_dec))
        self.w_clas.append(K.get_value(self.w_cla))
            


class MyCallback_low_high(callbacks.Callback):
    def __init__(self, w_dec, w_cla, n_epoch,l_period):
        
        self.n_epoch = n_epoch
        self.w_dec = w_dec
        self.w_cla = w_cla
        
        self.w_clas = []
        self.w_decs = []
        
        self.l_period = l_period * 2
        self.current = 0
        
    def on_epoch_end(self, epoch, logs={}):
        
        if (self.current <= (self.l_period/2)):
            self.current = self.current + 1
            K.set_value(self.w_cla, 0)
            K.set_value(self.w_dec, 1)
        else:
            self.current = self.current + 1
            K.set_value(self.w_cla, 1)
            K.set_value(self.w_dec, 0)
            
        if (self.current == self.l_period):
            self.current = 0

            
        self.w_decs.append(K.get_value(self.w_dec))
        self.w_clas.append(K.get_value(self.w_cla))


def gen_sigm_seq(plateau,k,epoche,period):
    
    points = int(epoche/(2*period))+1
    res = []
    for i in range(period):
        pos = gen_bell(plateau,k,points,True)
        res = res + pos

    return(res)
        
def gen_bell(plateau,k,points,sign):
    
    fir = gen_sigm(plateau,k,points,sign)
    sec = gen_sigm(plateau,k,points,not sign)

    return (fir+sec)
    
    

def gen_sigm(plateau,k,points,sign):
    samples = np.linspace(-plateau/2,plateau/2,points)

    res = []
    for i in samples:
        y = 1/(1+math.pow(math.e,-k*i))
        if (sign==False):
            res.append(-y +1 )    
        else:
            res.append(y)
    return(list(res))