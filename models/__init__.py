from keras.layers import Input, Dense,Conv2D
from keras.models import Sequential
import keras


def get_model_from_args(args,hidden_units=None):
    if args.model=='fcshallow':
        model = get_fullconnected_shallow(args,hidden_units)
    elif args.model=='fcdeep':
        model = get_fullconnected_deep(args)
    elif args.model=='convshallow':
        model = get_convnet_shallow(args)
    elif args.model=='convdeep':
        model = get_convnet_deep(args)
    elif args.model=='capsshallow':
        model = get_caps_shallow(args)  
    else:
        model = get_caps_deep(args) 
    return model
def get_fullconnected_shallow(args,hidden_units):
    input_shape = (32*32,)
    model = Sequential()
    model.add(Dense(hidden_units,input_shape=input_shape,activation="relu"))
    model.add(Dense(3,activation="softmax"))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(args.lr),metrics=["accuracy"])
    return model

def get_fullconnected_deep(args):
    input_shape = (32,32,1)
    

def get_convnet_shallow(args):
    input_shape = (32,32,1)
def get_convnet_deep(args):
    input_shape = (32,32,1)
    
def get_caps_shallow(args):
    input_shape = (32,32,1)
    
def get_caps_deep(args):
    input_shape = (32,32,1)
    

    