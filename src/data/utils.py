import os
import numpy as np
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


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)