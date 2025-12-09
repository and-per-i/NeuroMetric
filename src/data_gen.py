import cv2
import mediapipe as mp
import numpy as np
import os
import config
from anomaly_utils import AnomalyInjector

def get_kinematic_features(landmarks, tracked_indices):
    """
    Estrae le feature spaziali (Posizione) normalizzate sul naso.
    Restituisce anche la posizione GREZZA del naso per il calcolo globale.
    """
    all_lms = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # 1. Coordinate Naso (per ancoraggio e per velocità globale)
    nose_tip = all_lms[1]
    
    # 2. Ancoraggio (Sottrazione naso)
    centered_lms = all_lms - nose_tip
    
    # 3. Scala (Distanza occhi)
    left_eye = all_lms[33]
    right_eye = all_lms[263]
    dist_eyes = np.linalg.norm(right_eye - left_eye)
    scale_factor = dist_eyes if dist_eyes > 0 else 1.0
    
    normalized_lms = centered_lms / scale_factor
    final_features = normalized_lms[tracked_indices]
    
    return final_features, nose_tip, scale_factor

def estrai_punti_da_video(video_path):
    """
    V3.2: Estrae [Posizione Locale, Velocità Locale, VELOCITÀ TESTA].
    """
    print(f"Processing V3.2: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    sequence = []
    prev_local_lms = None
    prev_nose_pos = None
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # 1. Ottieni feature locali e dati globali
            current_local_lms, current_nose, scale = get_kinematic_features(
                results.multi_face_landmarks[0].landmark, 
                config.TRACKED_LANDMARKS
            )
            
            # 2. Calcola Velocità LOCALE (Micro-espressioni)
            if prev_local_lms is not None:
                velocity_local = current_local_lms - prev_local_lms
            else:
                velocity_local = np.zeros_like(current_local_lms)
            
            # 3. Calcola Velocità GLOBALE TESTA (Tic collo/Dondolio)
            # La normalizziamo con la scala per renderla indipendente dallo zoom
            if prev_nose_pos is not None:
                velocity_head = ((current_nose - prev_nose_pos) / scale) * 100.0
            else:
                velocity_head = np.array([0.0, 0.0])
                
            # Aggiorna stati precedenti
            prev_local_lms = current_local_lms.copy()
            prev_nose_pos = current_nose.copy()
            
            # 4. Costruisci vettore: [Pos_Loc(flat), Vel_Loc(flat), Vel_Head(2)]
            frame_vector = np.concatenate([
                current_local_lms.flatten(), 
                velocity_local.flatten(),
                velocity_head.flatten()
            ])
            
            sequence.append(frame_vector)
            
    cap.release()
    return np.array(sequence, dtype=np.float32)

def recalculate_velocity_v32(augmented_pos_3d, original_full_window):
    """
    Ricalcola la velocità locale dopo l'augmentation, ma MANTIENE la velocità della testa originale.
    """
    seq_len = augmented_pos_3d.shape[0]
    
    # 1. Appiattisci le nuove posizioni
    new_pos_flat = augmented_pos_3d.reshape(seq_len, -1)
    
    # 2. Calcola nuove velocità locali
    new_vel_local = np.zeros_like(new_pos_flat)
    new_vel_local[1:] = new_pos_flat[1:] - new_pos_flat[:-1]
    
    # 3. Recupera la velocità della testa originale (ultime 2 colonne)
    # Assumiamo che il tic facciale non cambi il movimento del collo esistente nel video base
    original_head_vel = original_full_window[:, -2:]
    
    # 4. Concatena tutto
    new_full_seq = []
    for i in range(seq_len):
        row = np.concatenate([
            new_pos_flat[i], 
            new_vel_local[i], 
            original_head_vel[i]
        ])
        new_full_seq.append(row)
        
    return np.array(new_full_seq, dtype=np.float32)

def genera_dataset_da_video():
    """Genera dataset Training V3.2"""
    if not os.path.exists(config.RAW_DATA_DIR):
        print("ERRORE: Cartella video non trovata!")
        return np.array([]), np.array([])

    X_list = []
    y_list = []
    
    injector = AnomalyInjector(fps=30)
    files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(".mp4")]
    
    if not files: return np.array([]), np.array([])

    for video_file in files:
        path = os.path.join(config.RAW_DATA_DIR, video_file)
        seq_reale = estrai_punti_da_video(path)
        
        # Check dimensioni (deve essere INPUT_SIZE definita in config)
        if seq_reale.shape[1] != config.INPUT_SIZE:
            continue
        if len(seq_reale) < config.SEQUENCE_LENGTH: continue

        # Calcolo indici per slicing
        # Struttura: [POS (N*2) | VEL (N*2) | HEAD (2)]
        num_landmarks = (config.INPUT_SIZE - 2) // 4
        split_pos_end = num_landmarks * 2 

        for i in range(0, len(seq_reale) - config.SEQUENCE_LENGTH, 15):
            window_full = seq_reale[i : i + config.SEQUENCE_LENGTH]
            
            # --- CLASSE 0: SANO ---
            X_list.append(window_full)
            y_list.append(0) 
            
            # Preparazione dati per augmentation (solo posizione)
            window_pos_flat = window_full[:, :split_pos_end]
            window_pos_3d = window_pos_flat.reshape(config.SEQUENCE_LENGTH, num_landmarks, 2)

            # --- CLASSE 1: TREMORE ---
            aug = injector.add_tremor(window_pos_3d.copy(), freq=np.random.uniform(4.0, 6.0))
            X_list.append(recalculate_velocity_v32(aug, window_full))
            y_list.append(1)

            # --- CLASSE 2: TIC ---
            aug = injector.add_tic(window_pos_3d.copy(), amplitude=np.random.uniform(0.04, 0.08))
            X_list.append(recalculate_velocity_v32(aug, window_full))
            y_list.append(2)

            # --- CLASSE 3: IPOMIMIA ---
            aug = injector.add_hypomimia(window_pos_3d.copy(), severity=np.random.uniform(0.6, 0.9))
            X_list.append(recalculate_velocity_v32(aug, window_full))
            y_list.append(3)

            # --- CLASSE 4: PARESI ---
            aug = injector.add_paresis(window_pos_3d.copy(), droop_factor=np.random.uniform(0.02, 0.05))
            X_list.append(recalculate_velocity_v32(aug, window_full))
            y_list.append(4)

            # --- CLASSE 5: DISCINESIA ---
            aug = injector.add_dyskinesia(window_pos_3d.copy(), intensity=np.random.uniform(0.03, 0.06))
            X_list.append(recalculate_velocity_v32(aug, window_full))
            y_list.append(5)

    if len(X_list) > 0:
        X_final = np.array(X_list, dtype=np.float32)
        y_final = np.array(y_list, dtype=np.int64)
    else:
        X_final = np.array([], dtype=np.float32)
        y_final = np.array([], dtype=np.int64)

    print(f"DATASET V3.2 GENERATO: {len(X_final)} campioni. Shape: {X_final.shape}")
    return X_final, y_final