import os
from sklearn.model_selection import GroupKFold


def split_train_val(filenames, labelnames, n_splits=5, is_train=True):
    groups = [os.path.dirname(f) for f in filenames]
    ys = [0] * len(filenames)
    
    gkf = GroupKFold(n_splits=n_splits)
    
    files, labels = [], []
    for i, (x, y) in enumerate(gkf.split(filenames, ys, groups)):
        if is_train:
            if i == 0:
                continue
                    
            files += list(filenames[y])
            labels += list(labelnames[y])
            
        else:
            files = list(filenames[y])
            labels = list(labelnames[y])
            break

    return files, labels