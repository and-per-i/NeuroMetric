import numpy as np
import random

class AnomalyInjector:
    def __init__(self, fps=30):
        self.fps = fps

    def add_tremor(self, sequence, freq=5.0, amplitude=0.015):
        """
        SINTOMO: Tremore a riposo (Parkinson).
        MATEMATICA: Onda sinusoidale continua sommata al segnale.
        MODIFICA V3.1: Adattato per array 3D (Tempo, Landm, Coord).
        """
        seq_new = sequence.copy()
        frames = len(sequence)
        t = np.arange(frames) / self.fps
        
        # Genera onda: A * sin(2 * pi * f * t). Shape (frames,)
        noise = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Applica il tremore all'asse Y (verticale) di TUTTI i landmark.
        # noise[:, np.newaxis] trasforma (frames,) in (frames, 1) per il broadcasting
        # contro seq_new[:, :, 1] che è (frames, num_landmarks).
        # seq_new[:, :, 1] è l'asse Y di TUTTI i landmark.
        seq_new[:, :, 1] += noise[:, np.newaxis]
        return seq_new

    def add_tic(self, sequence, duration_frames=4, amplitude=0.06):
        """
        SINTOMO: Tic nervoso / Tourette (AGGIORNAMENTO V3.2 - COMPLETO).
        Include sia TIC LOCALI (smorfie) che TIC GLOBALI (scatti del collo).
        """
        seq_new = sequence.copy()
        frames, num_landmarks, _ = sequence.shape
        
        # Scegliamo un punto casuale dove far avvenire il tic
        if frames > duration_frames + 2:
            start_tic = random.randint(1, frames - duration_frames - 1)
            end_tic = start_tic + duration_frames
            
            # --- A. TIC LOCALE (Smorfia: Occhi/Bocca) ---
            num_affected = random.randint(5, 12)
            affected_indices = np.random.choice(num_landmarks, size=num_affected, replace=False)
            dir_x_local = random.uniform(-0.8, 0.8)
            dir_y_local = random.uniform(-0.8, 0.8)

            # --- B. TIC GLOBALE (Scatto del Collo) ---
            # Simuliamo il collo spostando TUTTI i punti (Naso compreso)
            has_head_jerk = random.random() > 0.4  # 60% probabilità di scatto testa
            jerk_x = 0.0
            jerk_y = 0.0
            
            if has_head_jerk:
                # Lo scatto del collo è più ampio della smorfia
                jerk_amp = amplitude * random.uniform(1.5, 3.0) 
                jerk_x = random.choice([-1, 1]) * jerk_amp
                jerk_y = random.uniform(-0.5, 0.5) * jerk_amp

            # APPLICAZIONE (Moto Balistico)
            for t in range(start_tic, end_tic):
                progress = (t - start_tic) / duration_frames
                # Spike: movimento rapido andata e ritorno (sinusoide mezza onda)
                spike = np.sin(progress * np.pi) 
                
                # 1. Applica scatto testa a TUTTI i punti (Se attivo)
                if has_head_jerk:
                    seq_new[t, :, 0] += spike * jerk_x
                    seq_new[t, :, 1] += spike * jerk_y

                # 2. Applica smorfia solo ai punti locali
                for idx in affected_indices:
                    seq_new[t, idx, 0] += spike * amplitude * dir_x_local
                    seq_new[t, idx, 1] += spike * amplitude * dir_y_local
            
        return seq_new

    def add_hypomimia(self, sequence, severity=0.7):
        """
        SINTOMO: Ipomimia / Flat Affect (Depressione/Parkinson).
        MATEMATICA: Riduzione della varianza. Schiacciamo il segnale verso la sua media.
        """
        seq_new = sequence.copy()
        
        # Calcoliamo la posizione media
        mean_pos = np.mean(seq_new, axis=0)
        
        # Calcoliamo quanto si sta muovendo rispetto al centro
        movement = seq_new - mean_pos
        
        # Riduciamo il movimento
        damped_movement = movement * (1.0 - severity)
        
        return mean_pos + damped_movement

    def add_paresis(self, sequence, droop_factor=0.03):
        """
        SINTOMO: Paresi facciale / Asimmetria.
        MATEMATICA: Offset costante verso il basso (gravità) + immobilità quasi totale.
        MODIFICA V3.1: Adattato per array 3D (Tempo, Landm, Coord).
        """
        # Prima applichiamo un'ipomimia severa (già 3D compliant)
        seq_new = self.add_hypomimia(sequence, severity=0.9)
        
        # Poi aggiungiamo l'effetto "cadente" (drooping) alla Y di TUTTI i landmark.
        # Il droop_factor deve essere applicato a tutti i punti lungo il tempo.
        seq_new[:, :, 1] += droop_factor
        
        return seq_new

    def _generate_dyskinesia_wave(self, length):
        """
        Helper interno: Genera un'onda caotica a bassa frequenza.
        """
        t = np.arange(length) / self.fps
        y = np.zeros(length)

        # Componente Caotica a Bassa Frequenza (Movimento "Coreico")
        freqs = [0.5, 1.3, 2.1]
        weights = [0.6, 0.3, 0.1]
        
        for f, w in zip(freqs, weights):
            phase = random.uniform(0, 2 * np.pi) 
            variance = random.uniform(-0.1, 0.1) 
            y += w * np.sin(2 * np.pi * (f + variance) * t + phase)

        # Asimmetria / Offset (Deriva lenta)
        offset_wave = 0.5 * np.sin(2 * np.pi * 0.2 * t)
        y += offset_wave

        # Normalizzazione (-1 a 1)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        return y

    def add_dyskinesia(self, sequence, intensity=0.05):
        """
        SINTOMO: Discinesia Tardiva (TD).
        MATEMATICA: Onde lente, irregolari e asimmetriche (Random Walk fluido).
        """
        seq_new = sequence.copy()
        frames = len(sequence)
        
        # Generiamo le onde per X e Y separatamente
        wave_x = self._generate_dyskinesia_wave(frames)
        wave_y = self._generate_dyskinesia_wave(frames)

        for t in range(frames):
            seq_new[t, :, 0] += wave_x[t] * intensity
            seq_new[t, :, 1] += wave_y[t] * intensity
            
        return seq_new


def apply_random_anomaly(sequence):
    """
    Applica un'anomalia casuale con parametri RANDOMIZZATI (Stochastic Augmentation).
    """
    injector = AnomalyInjector(fps=30)
    
    # Lista delle patologie (assicurati che i nomi corrispondano alle classi del modello)
    symptoms_list = ['tremor', 'tic', 'hypomimia', 'paresis', 'dyskinesia']
    choice = random.choice(symptoms_list)
    
    # 1. TREMORE
    if choice == 'tremor':
        freq = random.uniform(3.5, 6.5) 
        amp = random.uniform(0.01, 0.025) 
        return injector.add_tremor(sequence, freq=freq, amplitude=amp), 1 # ID classe
        
    # 2. TIC
    elif choice == 'tic':
        durata = random.randint(2, 6) 
        amp = random.uniform(0.04, 0.08)
        return injector.add_tic(sequence, duration_frames=durata, amplitude=amp), 2
        
    # 3. IPOMIMIA
    elif choice == 'hypomimia':
        severity = random.uniform(0.4, 0.9)
        return injector.add_hypomimia(sequence, severity=severity), 3

    # 4. PARESI
    elif choice == 'paresis':
        droop = random.uniform(0.02, 0.05)
        return injector.add_paresis(sequence, droop_factor=droop), 4

    # 5. DISCINESIA
    elif choice == 'dyskinesia':
        amp = random.uniform(0.03, 0.06) 
        return injector.add_dyskinesia(sequence, intensity=amp), 5
    
    return sequence, 0 # Caso default (Sano)