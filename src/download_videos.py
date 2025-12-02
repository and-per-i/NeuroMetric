import os
import yt_dlp

# Configurazione Cartelle
RAW_DATA_PATH = "data/raw"
if not os.path.exists(RAW_DATA_PATH):
    os.makedirs(RAW_DATA_PATH)

# Lista di video selezionati (TED Talks e Interviste stabili)
# Questi video hanno inquadrature frontali e stabili, perfetti per il training.
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=UF8uR6Z6KLc", # Steve Jobs Stanford Address (Molto stabile)
    "https://www.youtube.com/watch?v=6Af6b_wyiwI", # Bill Gates TED (Faccia chiara)
    "https://www.youtube.com/watch?v=arj7oStGLkU", # Tim Urban TED (Espressivo)
    "https://www.youtube.com/watch?v=Ks-_Mh1QhMc", # Amy Cuddy TED
    "https://www.youtube.com/watch?v=c0KYU2j0TM4", # Brene Brown TED
]

def download_videos():
    print(f"--- INIZIO DOWNLOAD in {RAW_DATA_PATH} ---")
    print(f"Scaricamento di {len(VIDEO_URLS)} video di training...")
    
    ydl_opts = {
        # 'best[ext=mp4]': Scarica il miglior file singolo (audio+video già uniti).
        # Questo evita l'errore di FFmpeg mancante.
        'format': 'best[ext=mp4]', 
        
        # Salva col titolo del video nella cartella giusta
        'outtmpl': f'{RAW_DATA_PATH}/%(title)s.%(ext)s',
        
        # Opzioni per evitare blocchi
        'noplaylist': True,
        'ignoreerrors': True,
        'quiet': False,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(VIDEO_URLS)
        
    print(f"\n--- DOWNLOAD COMPLETATO ---")
    print(f"Controlla la cartella '{RAW_DATA_PATH}' per vedere i file.")

if __name__ == "__main__":
    download_videos()