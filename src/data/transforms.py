import albumentations as A

def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
    ])

def get_valid_transform():
    return A.Compose([
        A.Resize(512, 512)
    ])

def get_test_transform():
    return A.Compose([
        A.Resize(512, 512)
    ])