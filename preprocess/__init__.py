import pandas as pd 
import os
import numpy as np

def load_datasets(dataset_dir):
    train = pd.read_csv(os.path.join(dataset_dir,"train.amat"),sep=" ",dtype=np.float32,skiprows=1,header=None)
    test = pd.read_csv(os.path.join(dataset_dir,"test.amat"),sep=" ",dtype=np.float32,skiprows=1,header=None)
    valid = pd.read_csv(os.path.join(dataset_dir,"valid.amat"),sep=" ",dtype=np.float32,skiprows=1,header=None)
    
    train_set = train.values
    test_set  = test.values
    valid_set = valid.values

    return train_set,valid_set,test_set