import os
import glob
import subprocess

# --- CONFIGURAZIONE ---
TEST_VIDEO_DIR = "data/real_test_videos"

def run_all_tests():
    # Ottieni la cartella radice dello script corrente
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Costruisci il percorso completo della cartella dei video
    target_dir = os.path.join(project_root, TEST_VIDEO_DIR)
    
    # Cerca tutti i file .mp4
    video_files = glob.glob(os.path.join(target_dir, "*.mp4"))

    if not video_files:
        print(f"❌ ERRORE: Nessun file .mp4 trovato in {target_dir}")
        return
        
    print(f"\n🔥 Inizio analisi di {len(video_files)} video in sequenza...\n")

    for i, video_path in enumerate(video_files):
        # Prendiamo solo il nome file per i log
        filename = os.path.basename(video_path)
        
        print(f"🎬 [{i+1}/{len(video_files)}] Processando: {filename}...")
        
        # --- COMANDO DI ESECUZIONE CORRETTO PER IL TERMINALE WINDOWS/LINUX ---
        # Usiamo subprocess.run per lanciare lo script Python esterno
        try:
            # Comando: python src/main.py "percorso/completo/video.mp4"
            command = ["python", "src/main.py", video_path]
            
            # Subprocess lancia il comando nel terminale
            subprocess.run(command, check=True)
            
            print(f"✅ Fatto.")
            print("-" * 40)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore durante l'analisi di {filename}. Dettagli: {e}")
            print("-" * 40)

    print("\n🎉 TUTTO COMPLETATO! Controlla la cartella 'results'.")

if __name__ == "__main__":
    run_all_tests()