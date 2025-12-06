import torch
import os
import sys

# Se siamo su Colab, montiamo il drive e usiamo quel percorso
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Percorso assoluto su Colab
    BASE_DIR = "/content/drive/MyDrive/Neurometric"
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/neurometric_lstm_v2.pth")

else:
    # Percorso locale (sul tuo PC)
    RAW_DATA_DIR = "data/raw"
    MODEL_SAVE_PATH = "models/neurometric_lstm_v2.pth"

# --- MAPPA ANATOMICA AD ALTA DEFINIZIONE (INDICI MEDIAPIPE) ---
# Questi indici coprono il contorno completo, non solo i cardinali.

ANATOMY_MAP = {
    # Contorno Labbra (20 punti significativi presi dal loop esterno e interno)
    "BOCCA": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, # Labbro superiore
        146, 91, 181, 84, 17, 314, 405, 321, 375         # Labbro inferiore
    ],
    
    # Contorno Occhio Sinistro (16 punti)
    "OCCHIO_SX": [
        33, 246, 161, 160, 159, 158, 157, 173,
        133, 155, 154, 153, 145, 144, 163, 7
    ],

    # Contorno Occhio Destro (16 punti)
    "OCCHIO_DX": [
        362, 398, 384, 385, 386, 387, 388, 466,
        263, 249, 390, 373, 374, 380, 381, 382
    ],

    # Sopracciglia (5 punti per lato per vedere bene l'ipomimia)
    "SOPRACCIGLIA": [
        70, 63, 105, 66, 107,   # SX
        336, 296, 334, 293, 300 # DX
    ]
}

# Creiamo una lista piatta ordinata di tutti i punti da tracciare
TRACKED_LANDMARKS = []
for area in ANATOMY_MAP.values():
    TRACKED_LANDMARKS.extend(area)

# PARAMETRI DATI
SEQUENCE_LENGTH = 30  
# Input Size aumenta: ora sono (20+16+16+10) * 2 = 124 valori per frame
INPUT_SIZE = len(TRACKED_LANDMARKS) * 2 
NUM_CLASSES = 6       

# PARAMETRI TRAINING
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')