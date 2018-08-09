from train import get_train_args,train_from_args
from preprocess import load_datasets



def main():
    args = get_train_args()
    train_from_args(args)

if __name__ == '__main__':
    main()