from keras.layers import Input, Dense,Conv2D
from keras.models import Sequential


def get_model_from_args(args):
    if args.model=='fcshallow':
        model = get_fullconnected_shallow(args)
    elif args.model=='fcshallowsoft':
        model = get_fullconnected_shallow_softmax(args)
    elif args.model=='fcdeep':
        model = get_fullconnected_deep(args)
    elif args.model=='fcdeepsoft':
        model = get_fullconnected_deep_softmax(args)
    elif args.model=='convshallow':
        model = get_convnet_shallow(args)
    elif args.model=='convdeep':
        model = get_convnet_deep(args)
    elif args.model=='capsshallow':
        model = get_caps_shallow(args)  
    else:
        model = get_caps_deep(args) 
    return model
def get_fullconnected_shallow(args):
    input_shape = (32*32,)
    model = Sequential()
    model.add(Dense(2048,input_shape=input_shape))
    model.add(Dense(3,activation="sigmoid"))
    return model
    
def get_fullconnected_shallow_softmax(args):
    input_shape = (32,32,1)
    
def get_fullconnected_deep(args):
    input_shape = (32,32,1)
    
def get_fullconnected_deep_softmax(args):
    input_shape = (32,32,1)
    
def get_convnet_shallow(args):
    input_shape = (32,32,1)
def get_convnet_deep(args):
    input_shape = (32,32,1)
    
def get_caps_shallow(args):
    input_shape = (32,32,1)
    
def get_caps_deep(args):
    input_shape = (32,32,1)
    

    