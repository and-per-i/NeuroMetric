import torch

# PERCORSI
RAW_DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "neurometric_lstm.pth"

# --- MAPPA ANATOMICA (INDICI MEDIAPIPE) ---
ANATOMY_MAP = {
    "BOCCA": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        146, 91, 181, 84, 17, 314, 405, 321, 375
    ],
    "OCCHIO_SX": [
        33, 246, 161, 160, 159, 158, 157, 173,
        133, 155, 154, 153, 145, 144, 163, 7
    ],
    "OCCHIO_DX": [
        362, 398, 384, 385, 386, 387, 388, 466,
        263, 249, 390, 373, 374, 380, 381, 382
    ],
    "SOPRACCIGLIA": [
        70, 63, 105, 66, 107,
        336, 296, 334, 293, 300
    ]
}

TRACKED_LANDMARKS = []
for area in ANATOMY_MAP.values():
    TRACKED_LANDMARKS.extend(area)

# PARAMETRI DATI
SEQUENCE_LENGTH = 30  

# --- MODIFICA V3.2: INPUT SIZE AUMENTATA ---
# (N_Landmarks * 4) [Pos + Vel locale] + 2 [Velocità Testa Globale]
INPUT_SIZE = (len(TRACKED_LANDMARKS) * 4) + 2

NUM_CLASSES = 6       

# PARAMETRI TRAINING
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')