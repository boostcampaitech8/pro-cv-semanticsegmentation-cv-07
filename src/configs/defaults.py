import os


# 전역변수 설정
TRIAN_ROOT = "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/src/datasets/train"
SPLIT_FILE_ROOT = '/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/src/datasets/splits/fold_0'
TEST_IMAGE_ROOT = "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/src/datasets/test/DCM"
SAVED_DIR = "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/outputs/checkpoints"


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]
EXCLUDE_IMAGE_DIRS = {
    os.path.normpath("ID363"),
    os.path.normpath("ID487"),
}
EXCLUDE_LABEL_DIRS = {
    os.path.normpath("../outputs_json/ID363"),
    os.path.normpath("../outputs_json/ID487"),
}


# 훈련 하이퍼 파라미터 기본 값 설정
RANDOM_SEED = 21

BATCH_SIZE = 8
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_VAL = 2

LR = 1e-4
NUM_EPOCHS = 90
VAL_EVERY = 1
NUM_PATIENCE = 10