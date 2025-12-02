import numpy as np
import random

class AnomalyInjector:
    def __init__(self, fps=30):
        self.fps = fps

    def add_tremor(self, sequence, freq=5.0, amplitude=0.015):
        """
        SINTOMO: Tremore a riposo (Parkinson).
        MATEMATICA: Onda sinusoidale continua sommata al segnale.
        """
        seq_new = sequence.copy()
        frames = len(sequence)
        t = np.arange(frames) / self.fps
        
        # Genera onda: A * sin(2 * pi * f * t)
        noise = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Applica il tremore all'asse Y (verticale)
        seq_new[:, 1] += noise
        return seq_new

    def add_tic(self, sequence, duration_frames=4, amplitude=0.06):
        """
        SINTOMO: Tic nervoso / Tourette.
        MATEMATICA: Uno "spike" (scatto) improvviso e breve che interrompe il movimento.
        """
        seq_new = sequence.copy()
        frames = len(sequence)
        
        # Scegliamo un punto casuale dove far avvenire il tic
        # (evitiamo l'inizio e la fine estrema per non rompere la finestra)
        if frames > duration_frames + 2:
            start_tic = random.randint(1, frames - duration_frames - 1)
            end_tic = start_tic + duration_frames
            
            # Applichiamo lo scatto (es. palpebra che si chiude o bocca che scatta su)
            # Sottraiamo amplitude alla Y (in computer vision Y=0 è in alto)
            seq_new[start_tic:end_tic, 1] -= amplitude
            
        return seq_new

    def add_hypomimia(self, sequence, severity=0.7):
        """
        SINTOMO: Ipomimia / Flat Affect (Depressione/Parkinson).
        MATEMATICA: Riduzione della varianza. Schiacciamo il segnale verso la sua media.
        severity: 0.0 (normale) -> 1.0 (totalmente immobile)
        """
        seq_new = sequence.copy()
        
        # Calcoliamo la posizione media (il "centro" della faccia in quel secondo)
        mean_pos = np.mean(seq_new, axis=0)
        
        # Calcoliamo quanto si sta muovendo rispetto al centro
        movement = seq_new - mean_pos
        
        # Riduciamo il movimento
        # Se severity è 0.7, manteniamo solo il 30% del movimento originale
        damped_movement = movement * (1.0 - severity)
        
        return mean_pos + damped_movement

    def add_paresis(self, sequence, droop_factor=0.03):
        """
        SINTOMO: Paresi facciale / Asimmetria.
        MATEMATICA: Offset costante verso il basso (gravità) + immobilità quasi totale.
        """
        # Prima applichiamo un'ipomimia severa (il muscolo non risponde)
        seq_new = self.add_hypomimia(sequence, severity=0.9)
        
        # Poi aggiungiamo l'effetto "cadente" (drooping) alla Y
        # In MediaPipe Y aumenta scendendo, quindi + aggiunge gravità
        seq_new[:, 1] += droop_factor
        
        return seq_new

    def add_dyskinesia(self, sequence, freq=2.0, amplitude=0.04):
        """
        SINTOMO: Discinesia Tardiva (Effetto farmaci).
        MATEMATICA: Movimento ritmico lento (1-3Hz) della mandibola/bocca.
        Diverso dal tremore perché è più lento e ampio.
        """
        seq_new = sequence.copy()
        frames = len(sequence)
        t = np.arange(frames) / self.fps
        
        # Onda lenta (2Hz) e ampia
        movimento_ritmico = amplitude * np.sin(2 * np.pi * freq * t)
        
        # La discinesia spesso coinvolge movimenti laterali o verticali della bocca
        # Aggiungiamo sia su X che su Y per fare un movimento "rotatorio/masticatorio"
        seq_new[:, 0] += movimento_ritmico * 0.5 # Asse X (un po' meno)
        seq_new[:, 1] += movimento_ritmico       # Asse Y (principale)
        
        return seq_new

# Funzione helper per chiamare a caso un'anomalia (utile per il training misto)
# ... (Le classi AnomalyInjector restano uguali sopra) ...

# "Per evitare che il modello facesse overfitting su pattern fissi (es. imparare a memoria solo la frequenza di 5Hz), ho implementato una pipeline di Data Augmentation Stocastica. I parametri cinematici (frequenza, ampiezza, damping) vengono campionati da distribuzioni uniformi che coprono il range fisiologico reale della patologia."

def apply_random_anomaly(sequence, config):
    """
    Applica un'anomalia casuale con parametri RANDOMIZZATI per evitare l'overfitting.
    La rete non deve imparare a memoria, deve capire il concetto.
    """
    injector = AnomalyInjector(fps=30)
    
    # Lista delle patologie
    symptoms_list = ['tremor', 'tic', 'hypomimia', 'paresis', 'dyskinesia']
    choice = random.choice(symptoms_list)
    
    # 1. TREMORE (Parkinson)
    if choice == 'tremor':
        # Frequenza variabile (non sempre 5Hz, ma tra 3.5 e 6.5)
        freq = random.uniform(3.5, 6.5) 
        # Ampiezza variabile (a volte leggero, a volte forte)
        amp = random.uniform(0.01, 0.025) 
        return injector.add_tremor(sequence, freq=freq, amplitude=amp)
        
    # 2. TIC (Tourette)
    elif choice == 'tic':
        # Durata variabile dello scatto (più secco o più morbido)
        durata = random.randint(2, 6) 
        # Intensità dello scatto
        amp = random.uniform(0.04, 0.08)
        return injector.add_tic(sequence, duration_frames=durata, amplitude=amp)
        
    # 3. IPOMIMIA (Depressione)
    elif choice == 'hypomimia':
        # Gravità variabile: da "un po' spento" (0.4) a "faccia di pietra" (0.9)
        severity = random.uniform(0.4, 0.9)
        return injector.add_hypomimia(sequence, severity=severity)

    # 4. PARESI (Ictus)
    elif choice == 'paresis':
        # Quanto "cade" la faccia verso il basso
        droop = random.uniform(0.02, 0.05)
        return injector.add_paresis(sequence, droop_factor=droop)

    # 5. DISCINESIA (Farmaci)
    elif choice == 'dyskinesia':
        # Movimenti lenti e irregolari (tra 1Hz e 3Hz)
        freq = random.uniform(1.0, 3.0)
        amp = random.uniform(0.03, 0.06) # Movimenti ampi
        return injector.add_dyskinesia(sequence, freq=freq, amplitude=amp)
    
    return sequence