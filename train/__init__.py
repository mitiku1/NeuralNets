import argparse
from models import get_model_from_args
from preprocess import load_datasets
import keras

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset",required=True,help="Path to directory containing amat files")
    parser.add_argument("-b","--batch",required=32,help="Batch size for training")
    parser.add_argument("-e","--epoch",required=10,help="Epochs for training")
    parser.add_argument("-m","--model",default="fcshallow",help="Type of model to train",choices=['fcshallow', 'fcshallowsoft', 'fcdeep', 'fcdeepsoft', "convshallow", "convdeep","capsshallow","capsdeep"])
    return parser.parse_args()
def __train_model(model,args):
    input_shape = model.inputs[0].shape[1:]
    train, valid, test = load_datasets(args.dataset)
    
    train_x, train_y = train
    valid_x, valid_y = valid
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(1e-3),metrics=["accuracy"])
    model.fit(train_x,train_y,batch_size=64,epochs=30,validation_data=[valid_x,valid_y],verbose=1)
def train_from_args(args):
    model = get_model_from_args(args)
    __train_model(model,args)

