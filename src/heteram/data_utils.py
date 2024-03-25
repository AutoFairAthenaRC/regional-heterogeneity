import numpy as np

def split(X, y, train_size=0.8):
    indices = X.index.tolist()
    np.random.shuffle(indices)
    
    # data split
    train_size = int(train_size * len(X))
    
    X_train = X.iloc[indices[:train_size]].copy().reset_index(drop=True)
    Y_train = y.iloc[indices[:train_size]].copy().reset_index(drop=True)
    X_test = X.iloc[indices[train_size:]].copy().reset_index(drop=True)
    Y_test = y.iloc[indices[train_size:]].copy().reset_index(drop=True)
    
    return X_train, Y_train, X_test, Y_test
