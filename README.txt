# NeuroMetric: Analisi Cinetica dei Sintomi Neurologici (V3.0) 🧠

NeuroMetric è un progetto di tesi sviluppato per l'analisi e la classificazione automatica dei sintomi neurologici facciali (es. Tic, Tremore, Discinesia, Ipomimia).

Utilizzando una pipeline di visione artificiale basata su **MediaPipe** e una rete **LSTM (Long Short-Term Memory)**, il sistema processa sequenze video di pazienti per identificare e localizzare le anomalie del movimento in modo non invasivo.

---

## ✨ Caratteristiche Tecniche Principali (V3.0)

La versione 3.0 risolve le criticità di accuratezza introducendo il pre-processing cinematico e l'analisi dinamica.

### 1. Normalizzazione Cinematica (Stabilizzazione Testa)
Questa è la modifica strutturale chiave. Il sistema applica una trasformazione geometrica rigida (Traslazione, Rotazione, Scala) per **ancorare il volto** e **allineare gli occhi orizzontalmente**.
* **Obiettivo:** Isolare il movimento muscolare puro.
* **Beneficio:** Elimina i **falsi positivi di Paresi** e Ipomimia causati dalla rotazione o dall'inclinazione della testa del paziente.

### 2. Analisi Dinamica
Il vettore di input per l'LSTM è stato ampliato per includere la dinamica temporale.
* **Input V3.0:** Vettore di Posizione (Normalizzata) + **Vettore di Velocità** ($\Delta x, \Delta y$).
* **Beneficio:** Potenziamento della distinzione tra movimenti veloci e impulsivi (Tic) e la staticità anomala (Ipomimia).

### 3. Stack Tecnologico

* **Core Model:** PyTorch (LSTM)
* **Feature Extraction:** MediaPipe (Face Mesh), OpenCV
* **Linguaggio:** Python
* **Ambiente:** Google Colab

---

## 🚀 Prossimi Passi
Per eseguire il training, assicurarsi che `config.INPUT_SIZE` sia impostato su `N_LANDMARKS * 4` e che le credenziali GitHub siano state configurate in ambiente Colab.
