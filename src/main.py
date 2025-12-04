import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import os
import sys
import argparse

# Aggiunge la directory src al path per importare i moduli locali se lo script è lanciato dalla root
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Tenta l'importazione dei moduli
try:
    import config
    from model import NeuroLSTM
except ImportError:
    # Fallback se lanciato dalla root senza -m
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    import config
    from model import NeuroLSTM

# Mappa delle Classi (deve corrispondere a quella usata nel training)
CLASS_MAP = {
    0: "NORMALE",
    1: "TREMORE",
    2: "TIC",
    3: "IPOMIMIA",
    4: "PARESI",
    5: "DISCINESIA"
}

def main(video_filename):
    # --- 1. Configurazione Percorsi ---
    # Cerca il modello nella root del progetto
    project_root = os.path.dirname(current_dir) # Sale di un livello da src/
    model_path = os.path.join(project_root, "neurometric_lstm.pth")
    
    # Cerca il video nella cartella downloaded_videos
    video_path = os.path.join(project_root, "downloaded_videos", video_filename)
    
    # Nome file di output
    output_path = os.path.join(project_root, f"output_{video_filename}")

    # Controllo esistenza file
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello non trovato in: {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"❌ ERRORE: Video non trovato in: {video_path}")
        return

    # --- 2. Inizializzazione Modello ---
    device = torch.device('cpu') # Inferenza su CPU
    print(f"Loading model from: {model_path}")
    
    model = NeuroLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Modello caricato correttamente.")
    except Exception as e:
        print(f"❌ Errore caricamento pesi: {e}")
        return

    # --- 3. Setup MediaPipe ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- 4. Elaborazione Video ---
    cap = cv2.VideoCapture(video_path)
    
    # Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Buffer per la sequenza temporale (Sliding Window)
    buffer = deque(maxlen=config.SEQUENCE_LENGTH)
    
    print(f"▶️ Avvio analisi su: {video_filename}")
    print("Premi 'q' per interrompere anticipatamente.")

    landmark_idx = 0 # Indice del labbro superiore (usato nel training)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepara frame per MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Default visualizzazione
        status_text = "Analisi..."
        color = (255, 255, 0) # Giallo
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # 1. Estrazione Feature (Coordinate x,y del landmark target)
            lm = face_landmarks.landmark[landmark_idx]
            buffer.append([lm.x, lm.y])
            
            # Disegna punto di tracciamento
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # 2. Inferenza (solo se il buffer è pieno)
            if len(buffer) == config.SEQUENCE_LENGTH:
                input_tensor = torch.tensor([list(buffer)], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    
                    pred_class = predicted.item()
                    conf_val = confidence.item()

                # 3. Logica di Visualizzazione
                label = CLASS_MAP.get(pred_class, "Unknown")
                
                # Filtro per evitare falsi positivi con confidenza bassa
                if pred_class != 0 and conf_val > 0.75: 
                    status_text = f"RILEVATO: {label} ({conf_val:.0%})"
                    color = (0, 0, 255) # Rosso
                    # Box attorno alla faccia (approssimato)
                    cv2.rectangle(frame, (cx-100, cy-100), (cx+100, cy+100), color, 2)
                else:
                    status_text = "NORMALE"
                    color = (0, 255, 0) # Verde

        # Scrivi risultato sul frame
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Mostra e Salva
        cv2.imshow('NeuroMetric', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Analisi completata. Video salvato in: {output_path}")

if __name__ == "__main__":
    # Default video se non specificato
    default_video = "tremor_parkinsons_patient.mp4"
    
    # Se passi un argomento da riga di comando usa quello (es: python src/main.py tic_billie_eilish.mp4)
    if len(sys.argv) > 1:
        target_video = sys.argv[1]
    else:
        target_video = default_video
        
    main(target_video)