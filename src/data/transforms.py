import albumentations as A

def get_train_transform(size=512):
    return A.Compose([
        A.Resize(size, size),
    ])

def get_valid_transform(size=512):
    return A.Compose([
        A.Resize(size, size)
    ])

def get_test_transform(size=512):
    return A.Compose([
        A.Resize(size, size)
    ])