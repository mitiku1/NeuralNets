import argparse
from models import get_model_from_args
from preprocess import load_datasets
import keras
import json
from keras import backend as K
import gc

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset",type = str, required=True,help="Path to directory containing amat files")
    parser.add_argument("-b","--batch", type = int, default=32,help="Batch size for training")
    parser.add_argument("-e","--epoch",default=10, type = int,help="Epochs for training")
    parser.add_argument("-l","--lr",default=1e-3, type = float,help="learning rate for training")
    parser.add_argument("-i","--iterations",default=5, type = int,help="iteration to train model")
    parser.add_argument("-m","--model",default="fcshallow", type = str,help="Type of model to train",choices=['fcshallow','fcdeep',  "convshallow", "convdeep","capsshallow","capsdeep"])
    return parser.parse_args()

    
def train_from_args(args):
    hidden_units = [128, 256, 512, 1024, 2048, 4096, 8192]
    train_scores = dict()
    valid_scores = dict()
    for j in range(len(hidden_units)):
        ctrain_scores = []
        cvalid_scores = []
        for i in range(args.iterations):
            model = get_model_from_args(args,hidden_units[j])
            model.summary()
            input_shape = model.inputs[0].shape[1:]
            train, valid, test = load_datasets(args.dataset)
            
            train_x, train_y = train
            valid_x, valid_y = valid
            model.fit(train_x,train_y,batch_size=args.batch,epochs=args.epoch,validation_data=[valid_x,valid_y],verbose=1)
            valid_score = model.evaluate(valid_x,valid_y)
            train_score = model.evaluate(train_x,train_y)
            ctrain_scores.append(train_score)
            cvalid_scores.append(valid_score)
            print "Hidden units",hidden_units[j],"iteration",i
            print "train_score",train_score, "validation_score",valid_score
            del model
            K.clear_session()
            gc.collect()
        train_scores[hidden_units[j]] = ctrain_scores
        valid_scores[hidden_units[j]] = cvalid_scores

    with open('train_scores-%s.json' %(args.model), 'w+') as fp:
        json.dump(train_scores, fp)
    with open('valid_scores-%s.json' %(args.model), 'w+') as fp:
        json.dump(valid_scores, fp)
