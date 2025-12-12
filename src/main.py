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
PATIENCE_FRAMES = 30  
MIN_CLIP_FRAMES = 5   

# --- CONFIGURAZIONE VISIVA ---
VISUAL_HOLD_FRAMES = 8 

def create_session_structure(project_root, video_filename):
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(project_root, "results", f"{base_name}_{timestamp}")
    clips_dir = os.path.join(session_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    return session_dir, clips_dir

def generate_html_report(filepath, video_name, total_frames, fps, stats, events):
    """
    Genera un Report Clinico con Design Moderno e LIGHTBOX VIDEO (Loop & Blur).
    """
    date_str = datetime.datetime.now().strftime("%d/%m/%Y")
    
    # --- CALCOLO STATISTICHE DASHBOARD ---
    sorted_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)
    predom_label, predom_count = sorted_stats[0] if sorted_stats else ("N/A", 0)
    predom_perc = (predom_count / total_frames * 100) if total_frames > 0 else 0
    
    tic_count = stats.get("TIC", 0)
    tic_perc = (tic_count / total_frames * 100) if total_frames > 0 else 0
    tic_events_count = len([e for e in events if e['symptom'] == 'TIC'])

    disc_count = stats.get("DISCINESIA", 0)
    disc_perc = (disc_count / total_frames * 100) if total_frames > 0 else 0
    
    if events:
        avg_conf = sum(e['max_conf'] for e in events) / len(events)
    else:
        avg_conf = 0.0

    # --- GENERAZIONE RIGHE TABELLA ---
    rows_html = ""
    for idx, evt in enumerate(events):
        clip_rel_path = f"clips/{evt['filename']}"
        symptom_lower = evt['symptom'].lower()
        
        zones_list = evt['zone'].split(" + ")
        zones_html = "".join([f'<span class="zone-tag">{z}</span>' for z in zones_list])
        
        conf_class = "confidence-high" if evt['max_conf'] > 0.9 else "confidence-med"

        rows_html += f"""
        <tr>
            <td>#{idx+1:02d}</td>
            <td><span class="badge badge-{symptom_lower}">{evt['symptom']}</span></td>
            <td>{zones_html}</td>
            <td class="timing">{evt['start_time']:.1f}s <span style="font-size:0.8em; opacity:0.6">(+{evt['duration']:.1f}s)</span></td>
            <td><span class="{conf_class}">{evt['max_conf']:.1%}</span></td>
            <td>
                <div class="video-thumb" onclick="openModal('{clip_rel_path}')">
                    <video muted loop autoplay playsinline>
                        <source src="{clip_rel_path}" type="video/webm">
                        <source src="{clip_rel_path}" type="video/mp4">
                    </video>
                    <div class="play-overlay">🔍</div>
                </div>
            </td>
        </tr>"""

    # --- TEMPLATE HTML COMPLETO ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Report Clinico - {video_name}</title>
        <style>
            :root {{
                --bg-color: #f0f2f5;
                --card-bg: #ffffff;
                --text-primary: #1a1a1a;
                --text-secondary: #65676b;
                /* Colori Semantici */
                --color-normale-bg: #dcfce7; --color-normale-text: #166534;
                --color-tic-bg: #fff7ed;     --color-tic-text: #9a3412;
                --color-tremore-bg: #fef2f2; --color-tremore-text: #991b1b;
                --color-discinesia-bg: #eff6ff; --color-discinesia-text: #1e40af;
                --border-radius: 12px;
                --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
                --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
            }}

            body {{
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg-color);
                margin: 0; padding: 40px 20px; color: var(--text-primary);
            }}

            .container {{ max-width: 1100px; margin: 0 auto; }}

            header {{ margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
            h1 {{ margin: 0; font-size: 24px; color: #2c3e50; }}
            .subtitle {{ color: var(--text-secondary); margin-top: 5px; font-size: 14px; }}
            .meta-tag {{ background: #e2e8f0; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600; }}

            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: var(--card-bg); padding: 20px; border-radius: var(--border-radius); box-shadow: var(--shadow-sm); border: 1px solid #e5e7eb; transition: transform 0.2s; }}
            .stat-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
            .stat-label {{ font-size: 13px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
            .stat-value {{ font-size: 28px; font-weight: 700; margin: 10px 0; color: #333; }}
            .stat-sub {{ font-size: 13px; display: flex; align-items: center; gap: 5px; }}
            
            .card-success .stat-value {{ color: var(--color-normale-text); }}
            .card-warning .stat-value {{ color: var(--color-tic-text); }}

            .table-wrapper {{ background: var(--card-bg); border-radius: var(--border-radius); box-shadow: var(--shadow-sm); overflow: hidden; border: 1px solid #e5e7eb; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            th {{ background-color: #f8fafc; color: var(--text-secondary); font-weight: 600; font-size: 13px; text-transform: uppercase; padding: 16px 24px; border-bottom: 2px solid #e2e8f0; }}
            td {{ padding: 16px 24px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; font-size: 14px; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover {{ background-color: #f8fafc; }}

            .badge {{ padding: 6px 12px; border-radius: 50px; font-weight: 600; font-size: 12px; display: inline-block; }}
            .badge-normale {{ background: var(--color-normale-bg); color: var(--color-normale-text); }}
            .badge-tic {{ background: var(--color-tic-bg); color: var(--color-tic-text); }}
            .badge-tremore {{ background: var(--color-tremore-bg); color: var(--color-tremore-text); }}
            .badge-discinesia {{ background: var(--color-discinesia-bg); color: var(--color-discinesia-text); }}
            .badge-ipomimia {{ background: #f3e8ff; color: #6b21a8; }}
            .badge-paresi {{ background: #374151; color: #f9fafb; }}

            .video-thumb {{ 
                width: 140px; height: 80px; background-color: #000; border-radius: 8px; 
                overflow: hidden; display: flex; align-items: center; justify-content: center; 
                border: 1px solid #ddd; cursor: pointer; position: relative;
                transition: transform 0.2s;
            }}
            .video-thumb:hover {{ transform: scale(1.05); border-color: #3b82f6; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }}
            .video-thumb video {{ width: 100%; height: 100%; object-fit: cover; opacity: 0.9; }}
            .play-overlay {{
                position: absolute; color: white; font-size: 20px; pointer-events: none;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5); opacity: 0; transition: opacity 0.2s;
                background: rgba(0,0,0,0.4); padding: 8px; border-radius: 50%;
            }}
            .video-thumb:hover .play-overlay {{ opacity: 1; }}

            .confidence-high {{ color: var(--color-normale-text); font-weight: bold; }}
            .confidence-med {{ color: #d97706; font-weight: bold; }}
            .zone-tag {{ display: inline-block; background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px 2px 2px 0; }}
            .timing {{ font-family: 'Consolas', monospace; color: var(--text-secondary); }}

            /* --- MODAL (LIGHTBOX) STYLES --- */
            .modal-overlay {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0, 0, 0, 0.85); /* Sfondo scuro */
                backdrop-filter: blur(8px); /* Effetto Blur */
                z-index: 1000;
                display: none; /* Nascosto di default */
                justify-content: center; align-items: center;
                opacity: 0; transition: opacity 0.3s ease;
            }}
            .modal-overlay.active {{ display: flex; opacity: 1; }}
            
            .modal-content {{
                position: relative;
                max-width: 90%; max-height: 90%;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
                transform: scale(0.95); transition: transform 0.3s ease;
                background: black;
            }}
            .modal-overlay.active .modal-content {{ transform: scale(1); }}
            
            .modal-content video {{
                display: block; max-width: 100%; max-height: 85vh; margin: 0 auto;
            }}
            .close-btn {{
                position: absolute; top: 20px; right: 30px;
                color: white; font-size: 40px; cursor: pointer;
                z-index: 1001; text-shadow: 0 2px 10px rgba(0,0,0,0.5);
                transition: transform 0.2s;
            }}
            .close-btn:hover {{ transform: scale(1.1); color: #ef4444; }}
        </style>
    </head>
    <body>
        <div id="videoModal" class="modal-overlay" onclick="closeModal()">
            <div class="close-btn" onclick="closeModal()">&times;</div>
            <div class="modal-content" onclick="event.stopPropagation()">
                <video id="modalVideo" controls loop>
                    <source src="" type="video/mp4">
                    Il browser non supporta il tag video.
                </video>
            </div>
        </div>

        <div class="container">
            <header>
                <div><h1>Analisi NeuroMetrica</h1><div class="subtitle">File: {video_name} | Data: {date_str}</div></div>
                <div class="meta-tag">V3.2 AI Enhanced</div>
            </header>

            <div class="stats-grid">
                <div class="stat-card card-success"><div class="stat-label">Stato Predominante</div><div class="stat-value">{predom_label}</div><div class="stat-sub"><span>●</span> {predom_perc:.1f}% del tempo totale</div></div>
                <div class="stat-card card-warning"><div class="stat-label">Tic Rilevati</div><div class="stat-value">{tic_perc:.1f}%</div><div class="stat-sub">{tic_events_count} Eventi critici</div></div>
                <div class="stat-card"><div class="stat-label">Discinesia</div><div class="stat-value">{disc_perc:.1f}%</div><div class="stat-sub">Movimenti involontari</div></div>
                <div class="stat-card"><div class="stat-label">Affidabilità Media</div><div class="stat-value">{avg_conf:.1%}</div><div class="stat-sub">Qualità Rilevamento</div></div>
            </div>

            <div class="table-wrapper">
                <table>
                    <thead><tr><th width="5%">ID</th><th width="15%">Sintomo</th><th width="30%">Zona Interessata</th><th width="15%">Timing</th><th width="15%">Confidenza</th><th width="20%">Clip</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            
            <p style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 30px;">Report generato automaticamente da AI • Validare clinicamente</p>
        </div>

        <script>
            // Logica Modal
            function openModal(videoSrc) {{
                const modal = document.getElementById('videoModal');
                const video = document.getElementById('modalVideo');
                
                video.src = videoSrc;
                modal.classList.add('active');
                
                // Avvia video
                const playPromise = video.play();
                if (playPromise !== undefined) {{
                    playPromise.then(_ => {{
                        // Autoplay started
                    }}).catch(error => {{
                        console.log("Autoplay prevent from browser");
                    }});
                }}
            }}

            function closeModal() {{
                const modal = document.getElementById('videoModal');
                const video = document.getElementById('modalVideo');
                
                modal.classList.remove('active');
                setTimeout(() => {{
                    video.pause();
                    video.src = ""; 
                }}, 300); // Aspetta fine animazione CSS
            }}

            // Chiudi con ESC
            document.addEventListener('keydown', function(event) {{
                if (event.key === "Escape") {{
                    closeModal();
                }}
            }});
        </script>
    </body>
    </html>
    """
    with open(filepath, "w", encoding="utf-8") as f: f.write(html_content)

def find_dynamic_zone(buffer_array, anatomy_map):
    # OPTIMIZATION: Evitiamo conversioni inutili se è già array
    if not isinstance(buffer_array, np.ndarray):
        try:
            data = np.array(buffer_array, dtype=np.float32)
        except:
            return [], "N/A"
    else:
        data = buffer_array

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

    # --- VARIABILI PER LA PERSISTENZA VISIVA ---
    visual_cooldown = 0
    last_box_coords = None
    last_overlay_text = "NORMALE"
    last_overlay_color = (0, 255, 0)

    print(f"▶️ Analisi avviata...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- OTTIMIZZAZIONE PC LOCALE ---
        scale_percent = 50  
        new_width = int(frame.shape[1] * scale_percent / 100)
        new_height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (new_width, new_height))
        h, w, _ = frame.shape
        # -------------------------------
        
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
                velocity_head = ((current_nose - prev_nose_pos) / scale) * 100.0
            else:
                velocity_head = np.array([0.0, 0.0])

            prev_local_lms = current_local_lms.copy()
            prev_nose_pos = current_nose.copy()
            
            # CONCATENAZIONE V3.2: 250 Features
            frame_features_vector = np.concatenate([
                current_local_lms.flatten(), 
                velocity_local.flatten(),
                velocity_head.flatten()
            ])
            
            buffer.append(frame_features_vector)

            if len(buffer) == config.SEQUENCE_LENGTH:
                # --- OPTIMIZATION FIX: Conversione Diretta ---
                np_buffer = np.array(buffer, dtype=np.float32)
                input_tensor = torch.tensor(np_buffer, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(input_tensor)
                    probs = torch.softmax(out, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                pred_idx = pred.item()
                current_conf = conf.item()
                current_label = CLASS_MAP.get(pred_idx, "?")
                
                if pred_idx != 0 and current_conf > CONFIDENCE_THRESHOLD:
                    active_anomaly = True
                    # Per calcolare la zona, usiamo solo le feature di posizione
                    pos_feature_len = len(current_local_lms.flatten())
                    # Slice veloce su numpy
                    pos_only_buffer = np_buffer[:, :pos_feature_len]
                    zone_indices, current_zone_name = find_dynamic_zone(pos_only_buffer, config.ANATOMY_MAP)
                    
                    if zone_indices:
                        xs = [face_landmarks.landmark[i].x * w for i in zone_indices]
                        ys = [face_landmarks.landmark[i].y * h for i in zone_indices]
                        if xs and ys:
                            pad = 20
                            box_coords = (int(min(xs))-pad, int(min(ys))-pad, int(max(xs))+pad, int(max(ys))+pad)

        # --- LOGICA DI DISEGNO AVANZATA (VISUAL HOLD) ---
        if active_anomaly and box_coords:
            visual_cooldown = VISUAL_HOLD_FRAMES
            last_box_coords = box_coords
            last_overlay_text = f"{current_label} ({current_conf:.0%}) - {current_zone_name}"
            last_overlay_color = (0, 0, 255) # Rosso
        
        elif visual_cooldown > 0:
            visual_cooldown -= 1
        
        else:
            last_box_coords = None
            last_overlay_text = "NORMALE"
            last_overlay_color = (0, 255, 0) # Verde

        # Disegna
        if last_box_coords:
            cv2.rectangle(frame, 
                         (last_box_coords[0], last_box_coords[1]), 
                         (last_box_coords[2], last_box_coords[3]), 
                         last_overlay_color, 2)
            
            text_size, _ = cv2.getTextSize(last_overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, 
                         (last_box_coords[0], last_box_coords[1] - 25), 
                         (last_box_coords[0] + text_size[0], last_box_coords[1]), 
                         last_overlay_color, -1)
            
            cv2.putText(frame, last_overlay_text, 
                        (last_box_coords[0], last_box_coords[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "NORMALE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- GESTIONE CLIP ---
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
                
                if duration * fps > MIN_CLIP_FRAMES:
                    unique_zones = sorted(list(current_event["zones"]))
                    current_event["zone"] = " + ".join(unique_zones)
                    current_event["duration"] = duration
                    clip_events.append(current_event)
                    print(f"  🟢 STOP. Durata: {duration:.1f}s")
                else:
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
    print(f"\n✅ REPORT V3.2 GENERATO CON NUOVA GRAFICA: {html_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui l'inferenza del modello Neurometric su un video.")
    parser.add_argument("video", type=str, nargs='?', default="tic_billie_eilish.mp4", 
                        help="Nome del file video da analizzare (es. test.mp4).")
    args = parser.parse_args()
    main(args.video)