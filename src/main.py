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
    from data_gen import get_kinematic_features 
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import config
    from model import NeuroLSTM
    from data_gen import get_kinematic_features 

# --- MAPPA CLASSI ---
CLASS_MAP = {
    0: "NORMALE",
    1: "TREMORE",
    2: "TIC",
    3: "IPOMIMIA",
    4: "PARESI",
    5: "DISCINESIA"
}

# --- CONFIGURAZIONI (TUNING V3.2) ---
CONFIDENCE_THRESHOLD = 0.85
# Aumentiamo la pazienza per unire eventi vicini (es. Bocca -> Occhio)
PATIENCE_FRAMES = 30  
# Riduciamo la durata minima: i tic possono essere fulminei (0.2s)
MIN_CLIP_FRAMES = 5   

def create_session_structure(project_root, video_filename):
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(project_root, "results", f"{base_name}_{timestamp}")
    clips_dir = os.path.join(session_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    return session_dir, clips_dir

def generate_html_report(filepath, video_name, total_frames, fps, stats, events):
    """
    Genera un report HTML completo (Versione V3.0) con video embedded e stili CSS.
    """
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
            .NORMALE {{ background-color: #27ae60; }}
            video {{ border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Diagnosi: {video_name}</h1>
            {stat_html}
            <h3>Eventi Rilevati</h3>
            <table>
                <thead><tr><th>ID</th><th>Sintomo</th><th>Zona</th><th>Inizio</th><th>Durata</th><th>Confidenza</th><th>Clip</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </body>
    </html>
    """
    with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)

def find_dynamic_zone(buffer, anatomy_map):
    try:
        data = np.array(buffer, dtype=np.float32)
    except:
        return [], "N/A"

    if data.size == 0 or len(data.shape) < 2:
        return [], "Globale"

    max_score = -1
    best_zone_name = "Globale"
    best_indices = []
    current_idx = 0
    total_features = data.shape[1]

    for zone_name, indices in anatomy_map.items():
        n_points = len(indices)
        n_features = n_points * 2 
        
        if current_idx + n_features > total_features:
            break
            
        zone_data = data[:, current_idx : current_idx + n_features]
        movement_score = np.mean(np.std(zone_data, axis=0)) 
        
        if movement_score > max_score:
            max_score = movement_score
            best_zone_name = zone_name
            best_indices = indices
            
        current_idx += n_features
        
    return best_indices, best_zone_name

def main(video_filename):
    project_root = os.path.dirname(current_dir)
    possible_paths = [
        os.path.join(project_root, "data", "real_test_videos", video_filename),
        os.path.join(project_root, "data", "raw", video_filename),
        os.path.join(project_root, video_filename)
    ]
    video_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not video_path:
        print(f"❌ ERRORE: Video '{video_filename}' non trovato.")
        return

    model_path = os.path.join(project_root, config.MODEL_SAVE_PATH)
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello '{config.MODEL_SAVE_PATH}' mancante.")
        return

    session_dir, clips_dir = create_session_structure(project_root, video_filename)
    print(f"📂 Sessione: {session_dir}")

    device = torch.device('cpu')
    model = NeuroLSTM(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Modello V3.2 caricato (Input Size: {config.INPUT_SIZE}).")
    except Exception as e:
        print(f"❌ ERRORE CRITICO: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

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
    
    prev_local_lms = None 
    prev_nose_pos = None
    
    stats_counter = {v: 0 for k, v in CLASS_MAP.items()}
    clip_events = []
    is_recording_clip = False
    clip_writer = None
    current_event = {}
    patience = 0
    frame_idx = 0

    print(f"▶️ Analisi avviata...")

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
            
            # Feature Extraction V3.2
            current_local_lms, current_nose, scale = get_kinematic_features(
                face_landmarks.landmark, 
                config.TRACKED_LANDMARKS
            )
            
            current_local_lms = np.array(current_local_lms, dtype=np.float32)
            current_nose = np.array(current_nose, dtype=np.float32)

            if prev_local_lms is not None:
                velocity_local = current_local_lms - prev_local_lms
            else:
                velocity_local = np.zeros_like(current_local_lms)
            
            if prev_nose_pos is not None:
                velocity_head = (current_nose - prev_nose_pos) / scale
            else:
                velocity_head = np.array([0.0, 0.0])

            prev_local_lms = current_local_lms.copy()
            prev_nose_pos = current_nose.copy()
            
            frame_features_vector = np.concatenate([
                current_local_lms.flatten(), 
                velocity_local.flatten(),
                velocity_head.flatten()
            ])
            
            buffer.append(frame_features_vector)

            if len(buffer) == config.SEQUENCE_LENGTH:
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
                    pos_feature_len = len(current_local_lms.flatten())
                    pos_only_buffer = [f[:pos_feature_len] for f in buffer]
                    zone_indices, current_zone_name = find_dynamic_zone(pos_only_buffer, config.ANATOMY_MAP)
                    
                    if zone_indices:
                        xs = [face_landmarks.landmark[i].x * w for i in zone_indices]
                        ys = [face_landmarks.landmark[i].y * h for i in zone_indices]
                        if xs and ys:
                            pad = 20
                            box_coords = (int(min(xs))-pad, int(min(ys))-pad, int(max(xs))+pad, int(max(ys))+pad)

        if active_anomaly and box_coords:
            cv2.rectangle(frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"{current_label} ({current_conf:.0%}) - {current_zone_name}", 
                        (box_coords[0], box_coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NORMALE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- GESTIONE EVENTI MIGLIORATA ---
        if active_anomaly:
            patience = 0
            if not is_recording_clip:
                is_recording_clip = True
                clip_name = f"clip_{len(clip_events)+1}_{current_label}{ext}"
                clip_path = os.path.join(clips_dir, clip_name)
                clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
                
                current_event = {
                    "id": len(clip_events)+1, 
                    "symptom": current_label, 
                    "zones": {current_zone_name},
                    "start_time": frame_idx / fps, 
                    "max_conf": current_conf, 
                    "filename": clip_name
                }
                print(f"  🔴 REC: {current_label} in {current_zone_name}")
            else:
                # Accumuliamo le zone se il clip è già in registrazione
                current_event["zones"].add(current_zone_name)
                if current_conf > current_event.get("max_conf", 0): 
                    current_event["max_conf"] = current_conf

        elif is_recording_clip:
            patience += 1
            if patience > PATIENCE_FRAMES:
                is_recording_clip = False
                if clip_writer: clip_writer.release()
                clip_writer = None
                
                duration = (frame_idx / fps) - current_event["start_time"]
                
                # Controllo Durata Minima
                if duration * fps > MIN_CLIP_FRAMES:
                    unique_zones = sorted(list(current_event["zones"]))
                    current_event["zone"] = " + ".join(unique_zones)
                    current_event["duration"] = duration
                    
                    clip_events.append(current_event)
                    print(f"  🟢 STOP. Durata: {duration:.1f}s. Zone: {current_event['zone']}")
                else:
                    # DEBUG: Avvisa se stiamo buttando via dati
                    print(f"  🗑️ CLIP SCARTATA (Troppo breve: {duration:.2f}s) - {current_event['symptom']}")
                    try: os.remove(os.path.join(clips_dir, current_event['filename']))
                    except: pass
        
        stats_counter[current_label] = stats_counter.get(current_label, 0) + 1
        
        out_full.write(frame)
        if is_recording_clip and clip_writer: clip_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    if is_recording_clip and clip_writer: clip_writer.release()
    cap.release()
    out_full.release()
    cv2.destroyAllWindows()

    html_file = os.path.join(session_dir, "report_clinico.html")
    generate_html_report(html_file, os.path.basename(video_path), frame_idx, fps, stats_counter, clip_events)
    print(f"\n✅ REPORT V3.2 GENERATO: {html_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui l'inferenza del modello Neurometric su un video.")
    parser.add_argument("video", type=str, nargs='?', default="tic_billie_eilish.mp4", 
                        help="Nome del file video da analizzare (es. test.mp4).")
    args = parser.parse_args()
    main(args.video)