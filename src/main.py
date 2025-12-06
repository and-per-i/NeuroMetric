import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import os
import sys
import datetime
import argparse

# --- SETUP IMPORTAZIONI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import config
    from model import NeuroLSTM
    from data_gen import get_kinematic_features # NUOVO: Importa la funzione di normalizzazione cinematica
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import config
    from model import NeuroLSTM
    from data_gen import get_kinematic_features # NUOVO: Importa la funzione di normalizzazione cinematica

# --- MAPPA CLASSI ---
CLASS_MAP = {
    0: "NORMALE",
    1: "TREMORE",
    2: "TIC",
    3: "IPOMIMIA",
    4: "PARESI",
    5: "DISCINESIA"
}

# --- CONFIGURAZIONI ---
CONFIDENCE_THRESHOLD = 0.85
PATIENCE_FRAMES = 15
MIN_CLIP_FRAMES = 20

def create_session_structure(project_root, video_filename):
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Usa il nome del video come directory
    session_dir = os.path.join(project_root, "results", f"{base_name}_{timestamp}")
    clips_dir = os.path.join(session_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    return session_dir, clips_dir

def generate_html_report(filepath, video_name, total_frames, fps, stats, events):
    duration = total_frames / fps if fps > 0 else 0
    
    stat_html = ""
    for sintomo, count in stats.items():
        perc = (count / total_frames * 100) if total_frames > 0 else 0
        color = "#27ae60" if sintomo == "NORMALE" else ("#e74c3c" if perc > 10 else "#f39c12")
        stat_html += f"<p><strong>{sintomo}:</strong> <span style='color:{color}; font-weight:bold'>{perc:.1f}%</span></p>"

    rows_html = ""
    for evt in events:
        clip_rel_path = f"clips/{evt['filename']}"
        rows_html += f"""
        <tr>
            <td>{evt['id']}</td>
            <td><span class="badge {evt['symptom']}">{evt['symptom']}</span></td>
            <td>{evt['zone']}</td>
            <td>{evt['start_time']:.1f}s</td>
            <td>{evt['duration']:.1f}s</td>
            <td>{evt['max_conf']:.1%}</td>
            <td>
                <video width="200" controls loop preload="metadata">
                    <source src="{clip_rel_path}" type="video/webm">
                    <source src="{clip_rel_path}" type="video/mp4">
                    <source src="{clip_rel_path}" type="video/x-msvideo">
                    <p>Browser non supporta video. <a href="{clip_rel_path}">Scarica</a></p>
                </video>
            </td>
        </tr>"""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Report NeuroMetric - {video_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f4f4f9; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background-color: #34495e; color: white; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; font-size: 0.8em; }}
            .TREMORE {{ background-color: #e74c3c; }} .TIC {{ background-color: #f39c12; }}
            .IPOMIMIA {{ background-color: #9b59b6; }} .PARESI {{ background-color: #34495e; }}
            .DISCINESIA {{ background-color: #e67e22; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Diagnosi: {video_name}</h1>
            {stat_html}
            <h3>Eventi</h3>
            <table>
                <thead><tr><th>ID</th><th>Sintomo</th><th>Zona</th><th>Inizio</th><th>Durata</th><th>Confidenza</th><th>Clip</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </body>
    </html>
    """
    with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)

def get_active_zone_indices(pred_class, anatomy_map):
    # Logica per l'estrazione della zona è complessa e richiede aggiornamento completo
    # Manteniamo la logica esistente o semplifichiamo per il momento.
    # Visto che la V3.0 si basa sul movimento puro, find_dynamic_zone è più accurata.
    # Manteniamo la struttura esistente solo per coerenza, anche se find_dynamic_zone è la chiave.
    label = CLASS_MAP.get(pred_class, "IGNOTO")
    if label in ["TREMORE", "DISCINESIA"]:
        return anatomy_map["BOCCA"], "Regione Periorale"
    elif label == "TIC":
        return anatomy_map["OCCHIO_SX"] + anatomy_map["OCCHIO_DX"] + anatomy_map["SOPRACCIGLIA"], "Regione Oculare"
    elif label == "PARESI":
        return anatomy_map["BOCCA"] + anatomy_map["OCCHIO_SX"] + anatomy_map["OCCHIO_DX"], "Emivolto/Bocca"
    elif label == "IPOMIMIA":
        return config.TRACKED_LANDMARKS, "Intero Volto"
    return [], "Nessuna"

def find_dynamic_zone(buffer, anatomy_map):
    """
    Trova la zona anatomica con la maggiore deviazione standard (movimento).
    Il buffer contiene solo la POSIZIONE stabilizzata (la prima metà delle feature).
    """
    data = np.array(buffer)
    max_score = -1
    best_zone_name = "Globale"
    best_indices = []
    current_idx = 0
    
    # Calcola il numero totale di feature di POSIZIONE tracciate (N_Punti * 2)
    # Non possiamo usare config.INPUT_SIZE (che è N*4) qui.
    pos_feature_size = config.INPUT_SIZE // 2
    
    # Il buffer in ingresso QUI contiene solo la posizione, quindi la sua larghezza
    # sarà N_Punti * 2, anche se l'LSTM ne usa N*4.
    
    for zone_name, indices in anatomy_map.items():
        n_points = len(indices)
        n_features = n_points * 2 # Features di POSIZIONE per questa zona (x, y)
        
        # Estrai i dati di posizione della zona dal buffer
        zone_data = data[:, current_idx : current_idx + n_features]
        
        # Calcolo del punteggio di movimento: Deviazione standard (varianza) attraverso i frame (asse 0) e media sui punti
        movement_score = np.mean(np.std(zone_data, axis=0)) 
        
        if movement_score > max_score:
            max_score = movement_score
            best_zone_name = zone_name
            best_indices = indices
            
        current_idx += n_features
        
    return best_indices, best_zone_name

def main(video_filename):
    # 1. PERCORSI
    project_root = os.path.dirname(current_dir)
    possible_paths = [
        os.path.join(project_root, "data", "real_test_videos", video_filename),
        os.path.join(project_root, "data", "raw", video_filename), # Aggiunto percorso /data/raw
        os.path.join(project_root, video_filename)
    ]
    video_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not video_path:
        print(f"❌ ERRORE: Video '{video_filename}' non trovato.")
        return

    model_path = os.path.join(project_root, config.MODEL_SAVE_PATH) # Usa la variabile config
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello '{config.MODEL_SAVE_PATH}' mancante. Assicurati che il training sia finito.")
        return

    session_dir, clips_dir = create_session_structure(project_root, video_filename)
    print(f"📂 Sessione: {session_dir}")

    # 2. MODELLO
    device = torch.device('cpu')
    model = NeuroLSTM(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Modello V3.0 caricato con successo.")
    except RuntimeError as e:
        print(f"❌ ERRORE CRITICO: Dimensioni modello non corrispondenti. (V2.0 vs V3.0?)")
        print(f"Dettaglio: {e}")
        return

    # 3. VIDEO E PROCESSO
    cap = cv2.VideoCapture(video_path)
    success, first_frame = cap.read()
    if not success: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    h, w, _ = first_frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    ext = ".webm"
    
    out_full = cv2.VideoWriter(os.path.join(session_dir, f"FULL_{os.path.basename(video_filename)}{ext}"), 
                             fourcc, fps, (w, h))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    prev_landmarks_normalized = None # Memoria per calcolare la velocità nel main.py
    
    stats_counter = {v: 0 for k, v in CLASS_MAP.items()}
    clip_events = []
    is_recording_clip = False
    clip_writer = None
    current_event = {}
    patience = 0
    frame_idx = 0

    print(f"▶️ Analisi avviata (Input Size: {config.INPUT_SIZE})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        current_label = "NORMALE"
        current_conf = 0.0
        active_anomaly = False
        box_coords = None
        current_zone_name = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # --- AGGIORNAMENTO V3.0: FEATURE EXTRACTION ---
            # 1. Normalizzazione Cinematica (Posizione stabilizzata)
            current_lms_normalized = get_kinematic_features(
                face_landmarks.landmark, 
                config.TRACKED_LANDMARKS
            )
            
            # 2. Calcolo Velocità (Dinamica)
            if prev_landmarks_normalized is not None:
                velocity = current_lms_normalized - prev_landmarks_normalized
            else:
                velocity = np.zeros_like(current_lms_normalized)
                
            prev_landmarks_normalized = current_lms_normalized.copy()
            
            # 3. Vettore finale: [Posizione, Velocità]
            frame_features_vector = np.concatenate([
                current_lms_normalized.flatten(), 
                velocity.flatten()
            ])
            
            # Disegno dei landmark GREZZI (per visualizzazione)
            for idx in config.TRACKED_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (255, 255, 0), -1)
            
            buffer.append(frame_features_vector)

            if len(buffer) == config.SEQUENCE_LENGTH:
                # 4. Inferenza LSTM
                input_tensor = torch.tensor([list(buffer)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = model(input_tensor)
                    probs = torch.softmax(out, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                pred_idx = pred.item()
                current_conf = conf.item()

                current_label = CLASS_MAP.get(pred_idx, "?")
                
                if pred_idx != 0 and current_conf > CONFIDENCE_THRESHOLD:
                    active_anomaly = True
                    
                    # 5. Localizzazione Dinamica
                    # Estrai solo la Posizione (la prima metà) dal buffer per l'analisi spaziale
                    pos_only_buffer = [f[:config.INPUT_SIZE // 2] for f in buffer] 
                    
                    zone_indices, current_zone_name = find_dynamic_zone(pos_only_buffer, config.ANATOMY_MAP)
                    
                    if zone_indices:
                        # Calcola il riquadro di disegno usando i landmark grezzi per avere le coordinate originali
                        xs = [face_landmarks.landmark[i].x * w for i in zone_indices]
                        ys = [face_landmarks.landmark[i].y * h for i in zone_indices]
                        if xs and ys:
                            pad = 20
                            box_coords = (int(min(xs))-pad, int(min(ys))-pad, int(max(xs))+pad, int(max(ys))+pad)

        # Visualizzazione
        if active_anomaly and box_coords:
            cv2.rectangle(frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"{current_label} ({current_conf:.0%}) - {current_zone_name}", 
                        (box_coords[0], box_coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NORMALE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Gestione Clip
        if active_anomaly:
            # ... (logica di registrazione clip invariata)
            patience = 0
            if not is_recording_clip:
                is_recording_clip = True
                clip_name = f"clip_{len(clip_events)+1}_{current_label}{ext}"
                clip_path = os.path.join(clips_dir, clip_name)
                clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
                
                current_event = {
                    "id": len(clip_events)+1, "symptom": current_label, "zone": current_zone_name,
                    "start_time": frame_idx / fps, "max_conf": current_conf, "filename": clip_name
                }
                print(f"  🔴 REC: {current_label} in {current_zone_name}")
            else:
                if current_conf > current_event.get("max_conf", 0): 
                    current_event["max_conf"] = current_conf
                    current_event["zone"] = current_zone_name

        elif is_recording_clip:
            patience += 1
            if patience > PATIENCE_FRAMES:
                is_recording_clip = False
                if clip_writer: clip_writer.release()
                clip_writer = None
                
                duration = (frame_idx / fps) - current_event["start_time"]
                if duration * fps > MIN_CLIP_FRAMES:
                    current_event["duration"] = duration
                    clip_events.append(current_event)
                    print(f"  🟢 STOP. Durata: {duration:.1f}s")
                else:
                    try: os.remove(os.path.join(clips_dir, current_event['filename']))
                    except: pass
        
        # Aggiornamento contatore statistico
        stats_counter[current_label] = stats_counter.get(current_label, 0) + 1
        
        out_full.write(frame)
        if is_recording_clip and clip_writer: clip_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    # Cleanup finale
    if is_recording_clip and clip_writer: clip_writer.release()
    cap.release()
    out_full.release()
    cv2.destroyAllWindows()

    html_file = os.path.join(session_dir, "report_clinico.html")
    generate_html_report(html_file, os.path.basename(video_path), frame_idx, fps, stats_counter, clip_events)
    print(f"\n✅ REPORT GENERATO: {html_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui l'inferenza del modello Neurometric su un video.")
    parser.add_argument("video", type=str, nargs='?', default="tic_billie_eilish.mp4", 
                        help="Nome del file video da analizzare (es. test.mp4).")
    args = parser.parse_args()
    main(args.video)