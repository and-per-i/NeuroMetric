# NeuroMetric: Analisi Cinetica dei Sintomi Neurologici (V3.0) 🧠

NeuroMetric è un progetto di tesi sviluppato per l'analisi e la classificazione automatica dei sintomi neurologici facciali (es. Tic, Tremore, Discinesia, Ipomimia).

Utilizzando una pipeline di visione artificiale basata su **MediaPipe** e una rete **LSTM (Long Short-Term Memory)**, il sistema processa sequenze video di pazienti per identificare e localizzare le anomalie del movimento in modo non invasivo.

---

## ✨ Feature Implementate (V3.0)

L'attuale versione (V3.0) introduce tecniche avanzate di pre-processing per migliorare la robustezza del modello rispetto alle versioni precedenti.

### 1. Normalizzazione Cinematica (Stabilizzazione)
Per risolvere il problema del rumore causato dai movimenti della testa, è stata implementata una trasformazione geometrica rigida.
* **Tecnica:** Ancoraggio del volto e allineamento orizzontale degli occhi frame-by-frame.
* **Risultato:** Elimina i bias posizionali (es. falsi positivi di Ipomimia dovuti alla testa inclinata), isolando il movimento muscolare puro.

### 2. Analisi Dinamica (Velocity Features)
L'architettura LSTM ora riceve in input anche le derivate temporali, non solo le posizioni.
* **Input Vector:** Coordinate $(x, y)$ + **Velocità** $(\Delta x, \Delta y)$.
* **Risultato:** Il modello distingue meglio tra movimenti rapidi (Tic) e assenza di movimento (Ipomimia).

### 3. Stack Tecnologico
* **Framework:** PyTorch, MediaPipe, OpenCV
* **Modello:** LSTM (Long Short-Term Memory)
* **Ambiente:** Google Colab / Python