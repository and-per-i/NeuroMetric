import cv2
import mediapipe as mp
import numpy as np
import os
import config
from anomaly_utils import AnomalyInjector # <--- Importiamo la nostra "farmacia"
import random # <--- Aggiungiamo l'import per random

# Indice Landmark: 0 è il labbro superiore
LANDMARK_IDX = 0 

def estrai_punti_da_video(video_path):
    """Legge il video e trasforma i pixel in coordinate (x,y)"""
    print(f"Processing: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    sequence = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark[LANDMARK_IDX]
            sequence.append([lm.x, lm.y])
            
    cap.release()
    return np.array(sequence, dtype=np.float32)

def genera_dataset_da_video():
    """Genera dataset Training MULTI-CLASSE (0-5)"""
    if not os.path.exists(config.RAW_DATA_DIR):
        print("ERRORE: Cartella video non trovata!")
        return np.array([]), np.array([])

    X_list = []
    y_list = []
    
    # Inizializza l'iniettore
    injector = AnomalyInjector(fps=30) 

    files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(".mp4")]
    
    if not files:
        print("NESSUN VIDEO TROVATO. Esegui prima lo script di download!")
        return np.array([]), np.array([])

    for video_file in files:
        path = os.path.join(config.RAW_DATA_DIR, video_file)
        seq_reale = estrai_punti_da_video(path)
        
        if len(seq_reale) < config.SEQUENCE_LENGTH: continue

        # Taglia in finestre
        for i in range(0, len(seq_reale) - config.SEQUENCE_LENGTH, 15):
            window = seq_reale[i : i + config.SEQUENCE_LENGTH]
            
            # --- CLASSE 0: SANO ---
            X_list.append(window)
            y_list.append(0) 
            
            # --- CLASSE 1: TREMORE ---
            # Randomizziamo leggermente frequenza e ampiezza
            freq_tremor = random.uniform(3.5, 6.5)
            amp_tremor = random.uniform(0.01, 0.025)
            X_list.append(injector.add_tremor(window, freq=freq_tremor, amplitude=amp_tremor))
            y_list.append(1)

            # --- CLASSE 2: TIC ---
            # Randomizziamo durata e ampiezza
            durata_tic = random.randint(2, 6)
            amp_tic = random.uniform(0.04, 0.08)
            X_list.append(injector.add_tic(window, duration_frames=durata_tic, amplitude=amp_tic))
            y_list.append(2)

            # --- CLASSE 3: IPOMIMIA ---
            # Randomizziamo la gravità
            severity_hypo = random.uniform(0.4, 0.9)
            X_list.append(injector.add_hypomimia(window, severity=severity_hypo))
            y_list.append(3)

            # --- CLASSE 4: PARESI ---
            # Randomizziamo il fattore di "cedimento"
            droop_paresis = random.uniform(0.02, 0.05)
            X_list.append(injector.add_paresis(window, droop_factor=droop_paresis))
            y_list.append(4)

            # --- CLASSE 5: DISCINESIA ---
            # Randomizziamo frequenza e ampiezza
            freq_dys = random.uniform(1.0, 3.0)
            amp_dys = random.uniform(0.03, 0.06)
            X_list.append(injector.add_dyskinesia(window, freq=freq_dys, amplitude=amp_dys))
            y_list.append(5)

    X_final = np.array(X_list, dtype=np.float32)
    y_final = np.array(y_list, dtype=np.int64)
    
    print(f"DATASET GENERATO: {len(X_final)} campioni totali.")
    return X_final, y_final