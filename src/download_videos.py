import os
import yt_dlp

# Configurazione Cartelle
RAW_DATA_PATH = "data/raw"
if not os.path.exists(RAW_DATA_PATH):
    os.makedirs(RAW_DATA_PATH)

# Lista video YouTube (TED Talks - ottima qualità e stabilità)
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=UF8uR6Z6KLc", # Steve Jobs
    "https://www.youtube.com/watch?v=6Af6b_wyiwI", # Bill Gates
    "https://www.youtube.com/watch?v=arj7oStGLkU", # Tim Urban
    "https://www.youtube.com/watch?v=Ks-_Mh1QhMc", # Amy Cuddy
]

def download_videos():
    print(f"--- TENTATIVO DOWNLOAD YOUTUBE in {RAW_DATA_PATH} ---")
    
    ydl_opts = {
        # Cerchiamo il formato mp4 migliore sotto i 720p per evitare file enormi
        'format': 'best[ext=mp4][height<=720]/best[ext=mp4]',
        
        # Salva col titolo pulito
        'outtmpl': f'{RAW_DATA_PATH}/%(title)s.%(ext)s',
        
        # Opzioni anti-blocco
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': False,
        
        # Simula un browser reale per evitare il 403 Forbidden
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(VIDEO_URLS)
        
    # Verifica finale
    files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".mp4") and os.path.getsize(os.path.join(RAW_DATA_PATH, f)) > 0]
    if len(files) > 0:
        print(f"\n✅ SUCCESSO: Scaricati {len(files)} video validi.")
    else:
        print("\n❌ ATTENZIONE: Nessun video scaricato. YouTube sta bloccando l'IP di Colab.")
        print("Soluzione alternativa: Scarica i video sul tuo PC e caricali manualmente su Colab nella cartella data/raw.")

if __name__ == "__main__":
    download_videos()