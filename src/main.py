import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import os
import sys
import datetime
import argparse

# --- SETUP IMPORTAZIONI MODULI LOCALI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import config
    from model import NeuroLSTM
except ImportError:
    # Fallback se eseguito dalla root
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import config
    from model import NeuroLSTM

# --- MAPPA CLASSI ---
CLASS_MAP = {
    0: "NORMALE",
    1: "TREMORE",
    2: "TIC",
    3: "IPOMIMIA",
    4: "PARESI",
    5: "DISCINESIA"
}

# --- CONFIGURAZIONI UTILI ---
CONFIDENCE_THRESHOLD = 0.85
PATIENCE_FRAMES = 15    # Frame di attesa prima di chiudere una clip
MIN_CLIP_FRAMES = 20    # Durata minima per salvare una clip

# --- FUNZIONI DI UTILITÀ ---

def create_session_structure(project_root, video_filename):
    """Crea la struttura di cartelle per i risultati."""
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cartella principale della sessione
    session_dir = os.path.join(project_root, "results", f"{base_name}_{timestamp}")
    # Sottocartella per le clip
    clips_dir = os.path.join(session_dir, "clips")
    
    os.makedirs(clips_dir, exist_ok=True)
    return session_dir, clips_dir

def generate_html_report(filepath, video_name, total_frames, fps, stats, events):
    """Genera il report HTML interattivo."""
    duration = total_frames / fps if fps > 0 else 0
    
    # Genera HTML per le statistiche
    stat_html = ""
    for sintomo, count in stats.items():
        perc = (count / total_frames * 100) if total_frames > 0 else 0
        if count > 0 and sintomo != "NORMALE":
            color = "#e74c3c" if perc > 10 else "#f39c12"
        else:
            color = "#27ae60"
        stat_html += f"<p><strong>{sintomo}:</strong> <span style='color:{color}; font-weight:bold'>{perc:.1f}%</span></p>"

    # Genera HTML per la tabella eventi
    rows_html = ""
    for evt in events:
        clip_rel_path = f"clips/{evt['filename']}"
        rows_html += f"""
        <tr>
            <td>{evt['id']}</td>
            <td><span class="badge {evt['symptom']}">{evt['symptom']}</span></td>
            <td>{evt['start_time']:.1f}s</td>
            <td>{evt['duration']:.1f}s</td>
            <td>{evt['max_conf']:.1%}</td>
            <td>
                <video width="250" controls loop preload="metadata">
                    <source src="{clip_rel_path}" type="video/webm">
                    <source src="{clip_rel_path}" type="video/mp4"> Il browser non supporta il video.
                </video>
            </td>
        </tr>
        """

    # Template HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Report NeuroMetric - {video_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .stats-box {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background-color: #34495e; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; color: white; font-size: 0.9em; font-weight: bold; }}
            .TREMORE {{ background-color: #e74c3c; }} .TIC {{ background-color: #f39c12; }}
            .IPOMIMIA {{ background-color: #9b59b6; }} .PARESI {{ background-color: #34495e; }}
            .DISCINESIA {{ background-color: #e67e22; }}
            video {{ border-radius: 6px; background: #000; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NeuroMetric Diagnosis Report</h1>
            <p><strong>File:</strong> {video_name} | <strong>Durata:</strong> {duration:.1f}s</p>
            
            <div class="stats-box">
                <h3>Distribuzione Sintomi</h3>
                {stat_html}
            </div>

            <h3>Timeline Eventi</h3>
            <table>
                <thead>
                    <tr><th>ID</th><th>Sintomo</th><th>Inizio</th><th>Durata</th><th>Confidenza</th><th>Video</th></tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)


# --- FUNZIONE PRINCIPALE ---

def main(video_filename):
    # 1. GESTIONE PERCORSI
    project_root = os.path.dirname(current_dir)
    
    possible_paths = [
        os.path.join(project_root, "data", "real_test_videos", video_filename),
        os.path.join(project_root, "downloaded_videos", video_filename),
        os.path.join(project_root, video_filename)
    ]
    
    video_path = None
    for p in possible_paths:
        if os.path.exists(p):
            video_path = p
            break
            
    if not video_path:
        print(f"❌ ERRORE: Video '{video_filename}' non trovato.")
        return

    model_path = os.path.join(project_root, "neurometric_lstm.pth")
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello mancante in {model_path}")
        return

    # Crea output
    session_dir, clips_dir = create_session_structure(project_root, video_filename)
    print(f"📂 Output Sessione: {session_dir}")

    # 2. CARICAMENTO MODELLO
    device = torch.device('cpu')
    model = NeuroLSTM(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Errore caricamento pesi: {e}")
        return

    # 3. ANALISI VIDEO
    cap = cv2.VideoCapture(video_path)
    
    success, first_frame = cap.read()
    if not success: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    height, width, _ = first_frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    # --- SETUP CODEC WEBM (VP8) ---
    # Questo codec è supportato da Chrome/Edge/Firefox nativamente
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    
    # Video completo (webm)
    full_video_output = os.path.join(session_dir, f"FULL_{os.path.basename(video_filename)}.webm")
    out_full = cv2.VideoWriter(full_video_output, fourcc, fps, (width, height))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    
    stats_counter = {v: 0 for k, v in CLASS_MAP.items()}
    clip_events = []
    is_recording_clip = False
    clip_writer = None
    current_event = {}
    patience = 0
    frame_idx = 0
    current_symptom_for_clip = "NORMALE" # Variabile di appoggio

    print(f"▶️ Analisi in corso...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        current_label = "NORMALE"
        current_conf = 0.0
        active_anomaly = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark[0]
            buffer.append([lm.x, lm.y])
            
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

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
                    color = (0, 0, 255) # Rosso
                    box_size = 60
                    cv2.rectangle(frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), color, 2)
                    cv2.putText(frame, f"{current_label} {current_conf:.0%}", (cx - box_size, cy - box_size - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    current_label = "NORMALE"

        # --- GESTIONE CLIP ---
        if active_anomaly:
            patience = 0
            current_symptom_for_clip = current_label # Salva il nome per il file
            
            if not is_recording_clip:
                is_recording_clip = True
                # Nome file .webm per il browser
                clip_name = f"clip_{len(clip_events)+1}_{current_label}.webm"
                clip_path = os.path.join(clips_dir, clip_name)
                
                # Crea writer per la clip
                clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                
                current_event = {
                    "id": len(clip_events)+1,
                    "symptom": current_label,
                    "start_time": frame_idx / fps,
                    "max_conf": current_conf,
                    "filename": clip_name
                }
                print(f"  🔴 REC: {current_label}")
            else:
                if current_conf > current_event.get("max_conf", 0):
                    current_event["max_conf"] = current_conf

        elif is_recording_clip:
            patience += 1
            if patience > PATIENCE_FRAMES:
                is_recording_clip = False
                if clip_writer: clip_writer.release()
                clip_writer = None
                
                end_time = frame_idx / fps
                duration = end_time - current_event["start_time"]
                
                if duration * fps > MIN_CLIP_FRAMES:
                    current_event["duration"] = duration
                    clip_events.append(current_event)
                    print(f"  🟢 STOP. Durata: {duration:.1f}s")
                else:
                    try: os.remove(os.path.join(clips_dir, current_event['filename']))
                    except: pass
                    print("  ⚪ Clip ignorata")

        # Statistiche
        if current_label in stats_counter:
            stats_counter[current_label] += 1
        
        # Scrittura Frame
        cv2.putText(frame, f"Stato: {current_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if current_label=="NORMALE" else (0,0,255), 2)
        
        out_full.write(frame)
        if is_recording_clip and clip_writer:
            # Controllo dimensioni
            if frame.shape[:2] == (height, width):
                clip_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    if is_recording_clip and clip_writer: clip_writer.release()
    cap.release()
    out_full.release()
    cv2.destroyAllWindows()

    # Generazione Report HTML
    html_file = os.path.join(session_dir, "report_clinico.html")
    generate_html_report(html_file, video_filename, frame_idx, fps, stats_counter, clip_events)
    print(f"\n✅ REPORT GENERATO: {html_file}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tic_billie_eilish.mp4"
    main(target)