import os
import urllib.request

# Configurazione Cartelle
RAW_DATA_PATH = "data/raw"
if not os.path.exists(RAW_DATA_PATH):
    os.makedirs(RAW_DATA_PATH)

# Lista di link DIRETTI a file MP4 (Pexels - Stock Video Gratuiti)
# Questi link puntano direttamente al file, non a una pagina web.
VIDEO_URLS = [
    # Video 1: Uomo che parla in webcam (chiaro, stabile)
    ("video_man_talking.mp4", "https://videos.pexels.com/video-files/8317822/8317822-hd_1280_720_30fps.mp4"),
    
    # Video 2: Donna in videochiamata (ottimo contrasto)
    ("video_woman_call.mp4", "https://videos.pexels.com/video-files/7653229/7653229-hd_1280_720_25fps.mp4"),
    
    # Video 3: Uomo intervista (sfondo neutro)
    ("video_interview.mp4", "https://videos.pexels.com/video-files/4492610/4492610-hd_1280_720_25fps.mp4")
]

def download_videos():
    print(f"--- INIZIO DOWNLOAD DIRETTO (NO YOUTUBE) ---")
    
    for filename, url in VIDEO_URLS:
        file_path = os.path.join(RAW_DATA_PATH, filename)
        
        # Scarichiamo solo se non c'è già
        if not os.path.exists(file_path):
            print(f"Scaricamento: {filename}...")
            try:
                # urllib.request.urlretrieve è nativo di Python, non serve installare nulla
                urllib.request.urlretrieve(url, file_path)
                print(f"✅ Fatto: {filename}")
            except Exception as e:
                print(f"❌ Errore scaricamento {filename}: {e}")
        else:
            print(f"Gia presente: {filename}")

    # Verifica finale
    files = os.listdir(RAW_DATA_PATH)
    print(f"\n--- COMPLETATO: {len(files)} video pronti in {RAW_DATA_PATH} ---")

if __name__ == "__main__":
    download_videos()