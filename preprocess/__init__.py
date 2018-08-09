import pandas as pd 
import os
import numpy as np

def load_datasets(dataset_dir):
    """Loads babyai dataset 
    
    Arguments:
        dataset_dir {str} -- dataset directory containing train.amat,test.amat and validation.amat
    
    Returns:
        tuple -- arrays of train sets, test sets and validation sets
    """

    train = pd.read_csv(os.path.join(dataset_dir,"train.amat"),sep=" ",dtype=np.float32,skiprows=1,header=None)
    test = pd.read_csv(os.path.join(dataset_dir,"test.amat"),sep=" ",dtype=np.float32,skiprows=1,header=None)
    valid = pd.read_csv(os.path.join(dataset_dir,"valid.amat"),sep=" ",dtype=np.float32,skiprows=1,header=None)
    
    train_set = train.values
    test_set  = test.values
    valid_set = valid.values

    images_train,shapes_train = train_set[:,:1024],train_set[:,1024]
    images_test,shapes_test = test_set[:,:1024],test_set[:,1024]
    images_valid,shapes_valid = valid_set[:,:1024],valid_set[:,1024]

    shapes_train = np.eye(3)[shapes_train.astype(np.int32)]
    shapes_test = np.eye(3)[shapes_test.astype(np.int32)]
    shapes_valid = np.eye(3)[shapes_valid.astype(np.int32)]
    

    return [images_train,shapes_train],[images_valid,shapes_valid] ,[images_test, shapes_test]