import cv2
import mediapipe as mp
import numpy as np
import os
import config
from anomaly_utils import AnomalyInjector

def get_kinematic_features(landmarks, tracked_indices):
    """
    CORE V3.1: Normalizzazione Cinematica (Rotazione DISABILITATA).
    Trasforma i landmark grezzi in coordinate relative al naso,
    ma MANTIENE la rotazione della testa per rilevare i Tic del collo.
    """
    # Converti tutti i landmark in numpy array (468, 2)
    all_lms = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # --- STEP 1: ANCORAGGIO (Translation) ---
    # Sottraiamo il naso (Landmark 1) da tutti i punti.
    # Il volto è centrato su (0,0), ma i movimenti relativi restano.
    nose_tip = all_lms[1]
    centered_lms = all_lms - nose_tip
    
    # --- STEP 2: RADDRIZZAMENTO (Rotation) ---
    # MODIFICA V3.1: DISABILITATO.
    # Se il paziente ha un tic al collo (scatto laterale), dobbiamo vederlo!
    # Se raddrizziamo la testa, la velocity di quel movimento andrebbe a zero.
    
    # left_eye = centered_lms[33]
    # right_eye = centered_lms[263]
    # dY = right_eye[1] - left_eye[1]
    # dX = right_eye[0] - left_eye[0]
    # angle = np.arctan2(dY, dX)
    # c, s = np.cos(angle), np.sin(angle)
    # rotation_matrix = np.array(((c, s), (-s, c)))
    # rotated_lms = np.dot(centered_lms, rotation_matrix)
    
    rotated_lms = centered_lms # Nessuna rotazione applicata
    
    # --- STEP 3: SCALA (Scaling) ---
    # Normalizziamo in base alla distanza interpupillare per invarianza allo zoom.
    # Usiamo i punti originali per calcolare la distanza di riferimento.
    left_eye_orig = all_lms[33]
    right_eye_orig = all_lms[263]
    dist_eyes = np.linalg.norm(right_eye_orig - left_eye_orig)
    
    if dist_eyes > 0:
        normalized_lms = rotated_lms / dist_eyes
    else:
        normalized_lms = rotated_lms

    # Estraiamo solo i punti tracciati definiti nel config
    final_features = normalized_lms[tracked_indices]
    
    return final_features

def estrai_punti_da_video(video_path):
    """
    Legge il video, applica la normalizzazione V3.1 e calcola la VELOCITÀ.
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
    prev_landmarks = None # Memoria per calcolare la velocità
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # 1. Ottieni coordinate (senza correzione rotazione)
            current_lms = get_kinematic_features(
                results.multi_face_landmarks[0].landmark, 
                config.TRACKED_LANDMARKS
            )
            
            # 2. Calcola Velocità (Dinamica)
            # Ora se la testa ruota, velocity avrà valori alti!
            if prev_landmarks is not None:
                velocity = current_lms - prev_landmarks
            else:
                velocity = np.zeros_like(current_lms)
            
            prev_landmarks = current_lms.copy()
            
            # 3. Costruisci il vettore finale: [Posizioni, Velocità]
            frame_vector = np.concatenate([current_lms.flatten(), velocity.flatten()])
            
            sequence.append(frame_vector)
            
    cap.release()
    return np.array(sequence, dtype=np.float32)

def recalculate_velocity(position_sequence):
    """
    Ricalcola la velocità dopo aver iniettato anomalie sintetiche.
    Necessario perché add_tremor/add_tic modificano solo la posizione.
    """
    velocities = np.zeros_like(position_sequence)
    # vel[t] = pos[t] - pos[t-1]
    velocities[1:] = position_sequence[1:] - position_sequence[:-1]
    
    flattened_seq = []
    for i in range(len(position_sequence)):
        pos_flat = position_sequence[i].flatten()
        vel_flat = velocities[i].flatten()
        flattened_seq.append(np.concatenate([pos_flat, vel_flat]))
        
    return np.array(flattened_seq, dtype=np.float32)

def genera_dataset_da_video():
    """Genera dataset Training MULTI-CLASSE (0-5) con logica V3.1"""
    if not os.path.exists(config.RAW_DATA_DIR):
        print("ERRORE: Cartella video non trovata!")
        return np.array([]), np.array([])

    X_list = []
    y_list = []
    
    # Importante: assicurati che AnomalyInjector sia aggiornato (Step successivo)
    injector = AnomalyInjector(fps=30)
    files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(".mp4")]
    
    if not files:
        print("NESSUN VIDEO TROVATO.")
        return np.array([]), np.array([])

    for video_file in files:
        path = os.path.join(config.RAW_DATA_DIR, video_file)
        
        # seq_reale ora contiene i movimenti del collo (se presenti)
        seq_reale = estrai_punti_da_video(path)
        
        expected_dim = config.INPUT_SIZE 
        if seq_reale.shape[1] != expected_dim:
            print(f"⚠️ ERRORE DIMENSIONI: {video_file} ha {seq_reale.shape[1]} features, servono {expected_dim}.")
            continue
        
        if len(seq_reale) < config.SEQUENCE_LENGTH: continue

        num_landmarks = expected_dim // 4
        split_idx = num_landmarks * 2 

    for i in range(0, len(seq_reale) - config.SEQUENCE_LENGTH, 15):
            window_full = seq_reale[i : i + config.SEQUENCE_LENGTH]
            
            # --- CLASSE 0: SANO ---
            # Usiamo i dati reali così come sono
            X_list.append(window_full)
            y_list.append(0) 
            
            # --- PREPARAZIONE DATI SINTETICI ---
            window_pos_flat = window_full[:, :split_idx]
            # Reshape in 3D per l'injector: (Sequence, N_Landmarks, 2)
            window_pos_3d = window_pos_flat.reshape(config.SEQUENCE_LENGTH, num_landmarks, 2)

            # --- CLASSE 1: TREMORE (Fixed V3.1 - NO LOOP) ---
            aug_tremor = window_pos_3d.copy()
            freq_tremor = np.random.uniform(4.0, 6.0)
            amp_tremor = np.random.uniform(0.015, 0.025)
            # FIX: Passiamo la finestra 3D. anomaly_utils.add_tremor deve essere adattata.
            aug_tremor = injector.add_tremor(aug_tremor, freq=freq_tremor, amplitude=amp_tremor)
            X_list.append(recalculate_velocity(aug_tremor))
            y_list.append(1)

            # --- CLASSE 2: TIC (Fixed V3.1 - NO LOOP) ---
            aug_tic = window_pos_3d.copy()
            amp_tic = np.random.uniform(0.04, 0.08)
            # FIX: Passiamo la finestra 3D. La logica di gruppo è gestita in anomaly_utils.
            aug_tic = injector.add_tic(aug_tic, amplitude=amp_tic)
            X_list.append(recalculate_velocity(aug_tic))
            y_list.append(2)

            # --- CLASSE 3: IPOMIMIA (Già 3D Compliant) ---
            severity = np.random.uniform(0.6, 0.9)
            aug_hypo = injector.add_hypomimia(window_pos_3d, severity=severity)
            X_list.append(recalculate_velocity(aug_hypo))
            y_list.append(3)

            # --- CLASSE 4: PARESI (Fixed V3.1 - NO LOOP) ---
            aug_paresis = window_pos_3d.copy()
            droop = np.random.uniform(0.02, 0.05)
            # FIX: Passiamo la finestra 3D. anomaly_utils.add_paresis deve essere adattata.
            aug_paresis = injector.add_paresis(aug_paresis, droop_factor=droop)
            X_list.append(recalculate_velocity(aug_paresis))
            y_list.append(4)

            # --- CLASSE 5: DISCINESIA (Già 3D Compliant) ---
            intensity_dys = np.random.uniform(0.03, 0.06)
            aug_dys = injector.add_dyskinesia(window_pos_3d, intensity=intensity_dys)
            X_list.append(recalculate_velocity(aug_dys))
            y_list.append(5)

    print(f"DATASET GENERATO V3.1: {len(X_final)} campioni (Shape X: {X_final.shape}).")
    return X_final, y_final