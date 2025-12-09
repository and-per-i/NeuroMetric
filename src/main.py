import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import os
import sys
import datetime
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import config
    from model import NeuroLSTM
    # Importiamo la nuova funzione get_kinematic_features che restituisce 3 valori
    from data_gen import get_kinematic_features 
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import config
    from model import NeuroLSTM
    from data_gen import get_kinematic_features

CLASS_MAP = {
    0: "NORMALE", 1: "TREMORE", 2: "TIC", 
    3: "IPOMIMIA", 4: "PARESI", 5: "DISCINESIA"
}

CONFIDENCE_THRESHOLD = 0.85
PATIENCE_FRAMES = 15
MIN_CLIP_FRAMES = 20

def create_session_structure(project_root, video_filename):
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(project_root, "results", f"{base_name}_{timestamp}")
    clips_dir = os.path.join(session_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    return session_dir, clips_dir

def generate_html_report(filepath, video_name, total_frames, fps, stats, events):
    # (Mantieni la funzione identica a prima, è solo visualizzazione)
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
            <td><a href="{clip_rel_path}">Vedi Clip</a></td>
        </tr>"""

    html_content = f"""
    <!DOCTYPE html><html><head><title>Report {video_name}</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #333; color: white; }}
        .badge {{ padding: 4px; color: white; border-radius: 4px; }}
        .TREMORE {{ background: #e74c3c; }} .TIC {{ background: #f39c12; }}
        .IPOMIMIA {{ background: #8e44ad; }} .PARESI {{ background: #2c3e50; }}
        .DISCINESIA {{ background: #d35400; }}
    </style></head><body>
    <h1>Report: {video_name}</h1>{stat_html}
    <h3>Eventi Rilevati</h3><table><thead><tr><th>ID</th><th>Sintomo</th><th>Zona</th><th>Inizio</th><th>Durata</th><th>Conf</th><th>Link</th></tr></thead><tbody>{rows_html}</tbody></table>
    </body></html>"""
    with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)

def find_dynamic_zone(buffer, anatomy_map):
    # Analisi solo sulla parte posizionale (prima metà meno gli ultimi 2 elementi)
    data = np.array(buffer)
    # Rimuovi head velocity e local velocity per guardare solo posizione
    # Input size totale = Pos + Vel + Head
    # Pos size = (Input - 2) / 2
    pos_size = (config.INPUT_SIZE - 2) // 2
    
    max_score = -1
    best_zone_name = "Globale"
    best_indices = []
    current_idx = 0
    
    for zone_name, indices in anatomy_map.items():
        n_points = len(indices)
        n_features = n_points * 2
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
        os.path.join(project_root, "data", "raw", video_filename),
        os.path.join(project_root, video_filename)
    ]
    video_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not video_path:
        print(f"❌ Video non trovato: {video_filename}")
        return

    model_path = os.path.join(project_root, config.MODEL_SAVE_PATH)
    session_dir, clips_dir = create_session_structure(project_root, video_filename)
    
    # SETUP MODELLO V3.2
    device = torch.device('cpu')
    model = NeuroLSTM(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup Video Writer per Full Video annotato
    out_full = cv2.VideoWriter(
        os.path.join(session_dir, f"ANALYSIS_{os.path.basename(video_filename)}.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    prev_local_lms = None
    prev_nose_pos = None
    
    stats_counter = {v: 0 for k, v in CLASS_MAP.items()}
    clip_events = []
    is_recording = False
    clip_writer = None
    current_event = {}
    patience_counter = 0
    frame_idx = 0

    print(f"▶️ Analisi V3.2 avviata su: {video_filename}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        current_label = "NORMALE"
        current_conf = 0.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- FEATURE EXTRACTION V3.2 (INFERENZA) ---
            current_local, current_nose, scale = get_kinematic_features(landmarks, config.TRACKED_LANDMARKS)
            
            # 1. Vel Locale
            if prev_local_lms is not None:
                vel_local = current_local - prev_local_lms
            else:
                vel_local = np.zeros_like(current_local)
            
            # 2. Vel Testa (Globale)
            if prev_nose_pos is not None:
                vel_head = (current_nose - prev_nose_pos) / scale
            else:
                vel_head = np.array([0.0, 0.0])
                
            prev_local_lms = current_local.copy()
            prev_nose_pos = current_nose.copy()
            
            # 3. Vettore completo
            feat_vector = np.concatenate([
                current_local.flatten(), 
                vel_local.flatten(),
                vel_head.flatten()
            ])
            
            buffer.append(feat_vector)

            # --- INFERENZA ---
            if len(buffer) == config.SEQUENCE_LENGTH:
                input_tensor = torch.tensor([list(buffer)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = model(input_tensor)
                    probs = torch.softmax(out, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                pred_idx = pred.item()
                current_conf = conf.item()
                current_label = CLASS_MAP.get(pred_idx, "?")

                # Logica rilevamento (semplificata per leggibilità)
                if pred_idx != 0 and current_conf > CONFIDENCE_THRESHOLD:
                    # Gestione Clip
                    if not is_recording:
                        is_recording = True
                        clip_name = f"ev_{len(clip_events)}_{current_label}.mp4"
                        clip_writer = cv2.VideoWriter(os.path.join(clips_dir, clip_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        current_event = {"id": len(clip_events), "symptom": current_label, "start_time": frame_idx/fps, "max_conf": current_conf, "filename": clip_name, "zone": "Vedi Video"}
                    else:
                        patience_counter = 0
                        if current_conf > current_event["max_conf"]: current_event["max_conf"] = current_conf
                elif is_recording:
                    patience_counter += 1
                    if patience_counter > PATIENCE_FRAMES:
                        is_recording = False
                        if clip_writer: clip_writer.release()
                        current_event["duration"] = (frame_idx/fps) - current_event["start_time"]
                        if current_event["duration"] > 0.5: clip_events.append(current_event)

        # Disegno UI
        color = (0, 255, 0) if current_label == "NORMALE" else (0, 0, 255)
        cv2.putText(frame, f"{current_label} ({current_conf:.1%})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Disegno barra velocità testa (Debug visivo)
        if len(buffer) > 0:
            # Estraiamo l'ultima head velocity dal buffer
            hv = buffer[-1][-2:] # ultimi 2 valori
            mag = np.linalg.norm(hv) * 1000 # scaliamo per visibilità
            cv2.rectangle(frame, (20, 80), (20 + int(mag), 100), (255, 255, 0), -1)
            cv2.putText(frame, "Head Motion", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        out_full.write(frame)
        if is_recording and clip_writer: clip_writer.write(frame)
        
        stats_counter[current_label] += 1
        frame_idx += 1

    if is_recording and clip_writer: clip_writer.release()
    cap.release()
    out_full.release()
    
    generate_html_report(os.path.join(session_dir, "report.html"), video_filename, frame_idx, fps, stats_counter, clip_events)
    print("✅ Analisi completata.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, nargs='?', default="test.mp4")
    args = parser.parse_args()
    main(args.video)