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
            # --- MODIFICA CHIAVE: Loop su TUTTI i punti ---
            for idx in config.TRACKED_LANDMARKS:
                lm = results.multi_face_landmarks[0].landmark[idx]
                frame_data.extend([lm.x, lm.y]) # Appiattisce [x1, y1, x2, y2...]
            
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
            print(f"⚠️ ERRORE DIMENSIONI: Il video ha dato {seq_reale.shape[1]} feature, ma ne servono {expected_dim}.")
            continue
        
        if len(seq_reale) < config.SEQUENCE_LENGTH: continue

        for i in range(0, len(seq_reale) - config.SEQUENCE_LENGTH, 15):
            window = seq_reale[i : i + config.SEQUENCE_LENGTH]
            
            # --- CLASSE 0: SANO ---
            X_list.append(window)
            y_list.append(0) 
            
            # --- CLASSE 1: TREMORE ---
            freq_tremor = np.random.uniform(4.0, 6.0)
            X_list.append(injector.add_tremor(window, freq=freq_tremor))
            y_list.append(1)

            # --- CLASSE 2: TIC ---
            X_list.append(injector.add_tic(window))
            y_list.append(2)

            # --- CLASSE 3: IPOMIMIA ---
            severity = np.random.uniform(0.6, 0.9)
            X_list.append(injector.add_hypomimia(window, severity=severity))
            y_list.append(3)

            # --- CLASSE 4: PARESI ---
            X_list.append(injector.add_paresis(window))
            y_list.append(4)

            # --- CLASSE 5: DISCINESIA ---
            freq_dys = np.random.uniform(1.0, 3.0)
            X_list.append(injector.add_dyskinesia(window, freq=freq_dys))
            y_list.append(5)

    X_final = np.array(X_list, dtype=np.float32)
    y_final = np.array(y_list, dtype=np.int64)
    
    print(f"DATASET GENERATO: {len(X_final)} campioni.")
    return X_final, y_final