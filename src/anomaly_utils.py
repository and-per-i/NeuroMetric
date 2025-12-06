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
        if frames > duration_frames + 2:
            start_tic = random.randint(1, frames - duration_frames - 1)
            end_tic = start_tic + duration_frames
            
            # Applichiamo lo scatto (sottraiamo amplitude alla Y)
            seq_new[start_tic:end_tic, 1] -= amplitude
            
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
        """
        # Prima applichiamo un'ipomimia severa
        seq_new = self.add_hypomimia(sequence, severity=0.9)
        
        # Poi aggiungiamo l'effetto "cadente" (drooping) alla Y
        seq_new[:, 1] += droop_factor
        
        return seq_new

    def _generate_dyskinesia_wave(self, length):
        """
        Helper interno: Genera un'onda caotica a bassa frequenza.
        """
        t = np.arange(length) / self.fps
        y = np.zeros(length)

        # Componente Caotica a Bassa Frequenza (Movimento "Coreico")
        # Somma di 3 onde lente con fasi casuali
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
        # Qui ora chiamiamo correttamente il metodo della classe
        amp = random.uniform(0.03, 0.06) 
        return injector.add_dyskinesia(sequence, intensity=amp), 5
    
    return sequence, 0 # Caso default (Sano)