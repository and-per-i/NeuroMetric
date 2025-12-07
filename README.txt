🔬 Descrizione per GitHub: Progetto NeuroMetric (V3.0)
Titolo Breve: NeuroMetric: Analisi Cinetica dei Sintomi Neurologici tramite Modelli LSTM

Descrizione Estesa:

"NeuroMetric è un progetto di tesi sviluppato per l'analisi e la classificazione automatica dei sintomi neurologici facciali  (es. Tic, Tremore, Discinesia, Ipomimia).

Utilizzando una pipeline di visione artificiale basata su MediaPipe e una rete LSTM (Long Short-Term Memory) , il sistema processa sequenze video di pazienti per identificare e localizzare le anomalie del movimento.



Caratteristiche Principali (V3.0):
Normalizzazione Cinematica: Implementazione di una pre-elaborazione geometrica che stabilizza il volto (Traslazione e Rotazione) per isolare il movimento muscolare, eliminando i falsi positivi dovuti alla postura della testa.

Analisi Dinamica: Vettori di input composti da Posizione + Velocità per migliorare la distinzione tra movimenti rapidi (Tic) e staticità (Ipomimia).


Strumenti: Sviluppato in Python con PyTorch, NumPy, OpenCV e scikit-learn."
