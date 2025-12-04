import yt_dlp
import os

# --- INSERISCI I TUOI URL QUI ---
# Sostituisci "URL_DA_INSERIRE_QUI" con gli URL che hai trovato su YouTube.
video_list = [
    {
        "description": "tic_billie_eilish",
        "url": "https://www.youtube.com/watch?v=ATZpJlGYowQ"
    },
    {
        "description": "paresis_justin_bieber",
        "url": "https://www.youtube.com/watch?v=KbVVHJ4n1bQ"
    },
    {
        "description": "tremor_parkinsons_patient",
        "url": "https://www.youtube.com/watch?v=CqEwPqUO1Bw"
    }
]

# Imposta la cartella di destinazione per i video scaricati
DOWNLOAD_DIR = "downloaded_videos"

def download_videos(videos_to_download, download_path):
    """
    Scarica una lista di video da YouTube usando yt-dlp.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        print(f"Cartella '{download_path}' creata.")

    for video in videos_to_download:
        url = video["url"]
        description = video["description"]
        
        if "URL_DA_INSERIRE_QUI" in url:
            print(f"ATTENZIONE: Salto '{description}' perché l'URL non è stato inserito.")
            continue

        # Opzioni per yt-dlp
        # - f 'best': scarica la migliore qualità audio/video
        # - o: definisce il template per il nome del file di output
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(download_path, f'{description}.%(ext)s'),
        }

        try:
            print(f"Inizio download di: {description} ({url})")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Download di '{description}' completato.")
        except Exception as e:
            print(f"Errore durante il download di '{description}': {e}")
            
    print("\nProcesso di download terminato.")

if __name__ == "__main__":
    download_videos(video_list, DOWNLOAD_DIR)