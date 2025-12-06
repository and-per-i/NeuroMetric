import cv2
import mediapipe as mp
import numpy as np
import os
import config
from anomaly_utils import AnomalyInjector

def estrai_punti_da_video(video_path):
    """
    Legge il video ed estrae TUTTI i punti definiti in config.TRACKED_LANDMARKS
    Output Shape: (Frames, N_Punti * 2)
    """
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
            frame_data = []
            # Estrae i punti [x, y] per ogni landmark tracciato
            for idx in config.TRACKED_LANDMARKS:
                lm = results.multi_face_landmarks[0].landmark[idx]
                frame_data.extend([lm.x, lm.y]) 
            
            sequence.append(frame_data)
            
    cap.release()
    return np.array(sequence, dtype=np.float32)

def genera_dataset_da_video():
    """Genera dataset Training MULTI-CLASSE (0-5)"""
    if not os.path.exists(config.RAW_DATA_DIR):
        print("ERRORE: Cartella video non trovata!")
        return np.array([]), np.array([])

    X_list = []
    y_list = []
    
    injector = AnomalyInjector(fps=30)
    files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(".mp4")]
    
    if not files:
        print("NESSUN VIDEO TROVATO.")
        return np.array([]), np.array([])

    for video_file in files:
        path = os.path.join(config.RAW_DATA_DIR, video_file)
        seq_reale = estrai_punti_da_video(path)
        
        # Controllo dimensionale
        expected_dim = config.INPUT_SIZE
        if seq_reale.shape[1] != expected_dim:
            print(f"⚠️ ERRORE DIMENSIONI: {video_file} ha {seq_reale.shape[1]} features, servono {expected_dim}.")
            continue
        
        if len(seq_reale) < config.SEQUENCE_LENGTH: continue

        # Numero di landmark tracciati (ogni landmark ha x e y, quindi dividiamo per 2)
        num_landmarks = expected_dim // 2

        for i in range(0, len(seq_reale) - config.SEQUENCE_LENGTH, 15):
            window = seq_reale[i : i + config.SEQUENCE_LENGTH]
            
            # --- CLASSE 0: SANO ---
            X_list.append(window)
            y_list.append(0) 
            
            # Preparazione per l'iniezione anomalie: Reshape in (30, N_Punti, 2)
            # Questo ci permette di lavorare sui singoli punti se necessario
            window_3d = window.reshape(config.SEQUENCE_LENGTH, num_landmarks, 2)

            # --- CLASSE 1: TREMORE ---
            # Applichiamo il tremore a TUTTI i landmark tracciati
            aug_tremor = window_3d.copy()
            freq_tremor = np.random.uniform(4.0, 6.0)
            amp_tremor = np.random.uniform(0.015, 0.025)
            for k in range(num_landmarks):
                aug_tremor[:, k, :] = injector.add_tremor(aug_tremor[:, k, :], freq=freq_tremor, amplitude=amp_tremor)
            X_list.append(aug_tremor.reshape(config.SEQUENCE_LENGTH, -1)) # Flatten back
            y_list.append(1)

            # --- CLASSE 2: TIC ---
            aug_tic = window_3d.copy()
            amp_tic = np.random.uniform(0.04, 0.08)
            # Il tic colpisce spesso un punto specifico o tutti insieme. Qui lo applichiamo a tutti sincrono.
            for k in range(num_landmarks):
                aug_tic[:, k, :] = injector.add_tic(aug_tic[:, k, :], amplitude=amp_tic)
            X_list.append(aug_tic.reshape(config.SEQUENCE_LENGTH, -1))
            y_list.append(2)

            # --- CLASSE 3: IPOMIMIA ---
            # Hypomimia lavora bene su tutto l'array 3D direttamente (calcola la media globale)
            severity = np.random.uniform(0.6, 0.9)
            aug_hypo = injector.add_hypomimia(window_3d, severity=severity)
            X_list.append(aug_hypo.reshape(config.SEQUENCE_LENGTH, -1))
            y_list.append(3)

            # --- CLASSE 4: PARESI ---
            # Anche paresi può essere iterata per applicare il 'droop' (caduta) a tutti i punti
            aug_paresis = window_3d.copy()
            droop = np.random.uniform(0.02, 0.05)
            for k in range(num_landmarks):
                aug_paresis[:, k, :] = injector.add_paresis(aug_paresis[:, k, :], droop_factor=droop)
            X_list.append(aug_paresis.reshape(config.SEQUENCE_LENGTH, -1))
            y_list.append(4)

            # --- CLASSE 5: DISCINESIA (NUOVA LOGICA) ---
            # La funzione add_dyskinesia che abbiamo scritto nel nuovo anomaly_utils
            # gestisce GIA' l'input 3D internamente.
            # NOTA: Passiamo 'intensity', NON 'freq' (la frequenza è interna e complessa ora)
            intensity_dys = np.random.uniform(0.03, 0.06)
            aug_dys = injector.add_dyskinesia(window_3d, intensity=intensity_dys)
            X_list.append(aug_dys.reshape(config.SEQUENCE_LENGTH, -1))
            y_list.append(5)

    X_final = np.array(X_list, dtype=np.float32)
    y_final = np.array(y_list, dtype=np.int64)
    
    print(f"DATASET GENERATO: {len(X_final)} campioni (Shape X: {X_final.shape}).")
    return X_final, y_final