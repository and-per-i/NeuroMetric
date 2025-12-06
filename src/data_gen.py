import cv2
import mediapipe as mp
import numpy as np
import os
import config
from anomaly_utils import AnomalyInjector

def get_kinematic_features(landmarks, tracked_indices):
    """
    CORE V3.0: Normalizzazione Cinematica.
    Trasforma i landmark grezzi in coordinate relative al volto, 
    eliminando il movimento della testa (traslazione e rotazione).
    """
    # Converti tutti i landmark in numpy array (468, 2)
    all_lms = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # --- STEP 1: ANCORAGGIO (Translation) ---
    # Sottraiamo il naso (Landmark 1) da tutti i punti.
    # Ora il naso è sempre a (0,0), indipendentemente dalla posizione nella stanza.
    nose_tip = all_lms[1]
    centered_lms = all_lms - nose_tip
    
    # --- STEP 2: RADDRIZZAMENTO (Rotation) ---
    # Calcoliamo l'angolo tra gli occhi per annullare l'inclinazione della testa (Roll).
    # Questo elimina i falsi positivi di "Paresi" quando il paziente inclina la testa.
    # 33 = Angolo esterno occhio SX, 263 = Angolo esterno occhio DX
    left_eye = centered_lms[33]
    right_eye = centered_lms[263]
    
    # Calcolo angolo e matrice di rotazione inversa
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.arctan2(dY, dX)
    
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array(((c, s), (-s, c)))
    
    # Applichiamo la rotazione a tutti i punti
    rotated_lms = np.dot(centered_lms, rotation_matrix)
    
    # --- STEP 3: SCALA (Scaling) ---
    # Normalizziamo in base alla distanza interpupillare per invarianza allo zoom.
    dist_eyes = np.linalg.norm(right_eye - left_eye)
    if dist_eyes > 0:
        normalized_lms = rotated_lms / dist_eyes
    else:
        normalized_lms = rotated_lms

    # Estraiamo solo i punti tracciati definiti nel config
    final_features = normalized_lms[tracked_indices]
    
    return final_features

def estrai_punti_da_video(video_path):
    """
    Legge il video, applica la normalizzazione cinematica e calcola la VELOCITÀ.
    Output Shape: (Frames, N_Punti * 4) -> [Posizioni(x,y)..., Velocità(vx,vy)...]
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
    prev_landmarks = None # Memoria per calcolare la velocità (frame t - frame t-1)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # 1. Ottieni coordinate stabilizzate (Solo Posizione: N x 2)
            current_lms = get_kinematic_features(
                results.multi_face_landmarks[0].landmark, 
                config.TRACKED_LANDMARKS
            )
            
            # 2. Calcola Velocità (Dinamica)
            # La velocità è fondamentale per distinguere Tic (veloce) da Ipomimia (lenta)
            if prev_landmarks is not None:
                velocity = current_lms - prev_landmarks
            else:
                velocity = np.zeros_like(current_lms) # Primo frame: velocità 0
            
            prev_landmarks = current_lms.copy()
            
            # 3. Costruisci il vettore finale: [Tutte le Posizioni, Tutte le Velocità]
            # Flattening: trasforma matrice N x 2 in vettore lungo
            frame_vector = np.concatenate([current_lms.flatten(), velocity.flatten()])
            
            sequence.append(frame_vector)
            
    cap.release()
    return np.array(sequence, dtype=np.float32)

def recalculate_velocity(position_sequence):
    """
    Helper per ricalcolare la velocità dopo aver iniettato anomalie sintetiche.
    Se modifichiamo la posizione (es. aggiungiamo tremore), la velocità vecchia non vale più.
    Input: (Sequence_Len, N_Landmarks, 2)
    Output: (Sequence_Len, Input_Size) appiattito
    """
    velocities = np.zeros_like(position_sequence)
    # Calcolo differenza: vel[t] = pos[t] - pos[t-1]
    # Usiamo slicing numpy: dal secondo elemento in poi - dal primo al penultimo
    velocities[1:] = position_sequence[1:] - position_sequence[:-1]
    
    # Ricostruiamo il formato piatto [Posizione, Velocità] per ogni frame
    flattened_seq = []
    for i in range(len(position_sequence)):
        pos_flat = position_sequence[i].flatten()
        vel_flat = velocities[i].flatten()
        flattened_seq.append(np.concatenate([pos_flat, vel_flat]))
        
    return np.array(flattened_seq, dtype=np.float32)

def genera_dataset_da_video():
    """Genera dataset Training MULTI-CLASSE (0-5) con logica V3.0"""
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
        
        # seq_reale ora contiene [Posizione + Velocità]
        # Shape: (Frames, N_Landmarks * 4)
        seq_reale = estrai_punti_da_video(path)
        
        # Controllo dimensionale aggiornato
        expected_dim = config.INPUT_SIZE # Assicurati che sia N * 4 nel config!
        if seq_reale.shape[1] != expected_dim:
            print(f"⚠️ ERRORE DIMENSIONI: {video_file} ha {seq_reale.shape[1]} features, servono {expected_dim}.")
            continue
        
        if len(seq_reale) < config.SEQUENCE_LENGTH: continue

        # Numero di landmark tracciati
        # expected_dim è (N * 2 pos) + (N * 2 vel) = N * 4. Quindi N = dim / 4
        num_landmarks = expected_dim // 4
        # L'indice di split tra posizione e velocità è a metà del vettore
        split_idx = num_landmarks * 2 

        for i in range(0, len(seq_reale) - config.SEQUENCE_LENGTH, 15):
            window_full = seq_reale[i : i + config.SEQUENCE_LENGTH]
            
            # --- CLASSE 0: SANO ---
            # Usiamo i dati reali così come sono (già normalizzati e con velocità)
            X_list.append(window_full)
            y_list.append(0) 
            
            # --- PREPARAZIONE DATI SINTETICI ---
            # Per generare anomalie, dobbiamo lavorare solo sulla POSIZIONE.
            # La velocità deve essere ricalcolata DOPO aver aggiunto il sintomo.
            
            # Estraiamo solo la parte di posizione (prima metà delle colonne)
            window_pos_flat = window_full[:, :split_idx]
            # Reshape in 3D per l'injector: (Sequence, N_Landmarks, 2)
            window_pos_3d = window_pos_flat.reshape(config.SEQUENCE_LENGTH, num_landmarks, 2)

            # --- CLASSE 1: TREMORE ---
            aug_tremor = window_pos_3d.copy()
            freq_tremor = np.random.uniform(4.0, 6.0)
            amp_tremor = np.random.uniform(0.015, 0.025)
            for k in range(num_landmarks):
                aug_tremor[:, k, :] = injector.add_tremor(aug_tremor[:, k, :], freq=freq_tremor, amplitude=amp_tremor)
            # Ricalcola velocità e unisci
            X_list.append(recalculate_velocity(aug_tremor))
            y_list.append(1)

            # --- CLASSE 2: TIC ---
            aug_tic = window_pos_3d.copy()
            amp_tic = np.random.uniform(0.04, 0.08)
            for k in range(num_landmarks):
                aug_tic[:, k, :] = injector.add_tic(aug_tic[:, k, :], amplitude=amp_tic)
            X_list.append(recalculate_velocity(aug_tic))
            y_list.append(2)

            # --- CLASSE 3: IPOMIMIA ---
            severity = np.random.uniform(0.6, 0.9)
            aug_hypo = injector.add_hypomimia(window_pos_3d, severity=severity)
            X_list.append(recalculate_velocity(aug_hypo))
            y_list.append(3)

            # --- CLASSE 4: PARESI ---
            aug_paresis = window_pos_3d.copy()
            droop = np.random.uniform(0.02, 0.05)
            for k in range(num_landmarks):
                aug_paresis[:, k, :] = injector.add_paresis(aug_paresis[:, k, :], droop_factor=droop)
            X_list.append(recalculate_velocity(aug_paresis))
            y_list.append(4)

            # --- CLASSE 5: DISCINESIA ---
            intensity_dys = np.random.uniform(0.03, 0.06)
            aug_dys = injector.add_dyskinesia(window_pos_3d, intensity=intensity_dys)
            X_list.append(recalculate_velocity(aug_dys))
            y_list.append(5)

    X_final = np.array(X_list, dtype=np.float32)
    y_final = np.array(y_list, dtype=np.int64)
    
    print(f"DATASET GENERATO V3.0: {len(X_final)} campioni (Shape X: {X_final.shape}).")
    return X_final, y_final