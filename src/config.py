import torch

# PERCORSI
RAW_DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "neurometric_lstm.pth"

# PARAMETRI DATI
SEQUENCE_LENGTH = 30  # 30 Frame = 1 secondo
INPUT_SIZE = 2        # (x, y) per ogni punto
NUM_CLASSES = 6       # 0=Sano, 1=Tremore, 2=Tic, 3=Ipomimia, 4=Paresi, 5=Discinesia

# PARAMETRI TRAINING
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

# DISPOSITIVO
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')