# 🧠 NeuroMetric V3.2: Analisi Video Cinetica per la Diagnosi Neurologica

> **Sistema di Computer Vision "Markerless" basato su Deep Learning per la quantificazione e classificazione di anomalie motorie facciali.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-green)](https://google.github.io/mediapipe/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Overview
**NeuroMetric** è un framework di Intelligenza Artificiale sviluppato per supportare la diagnosi oggettiva di disturbi del movimento (*Movement Disorders*) partendo da semplici video RGB, senza l'uso di sensori indossabili.

Il sistema utilizza una pipeline ibrida che combina l'estrazione geometrica di feature tramite **MediaPipe** con un'analisi temporale profonda tramite reti neurali ricorrenti (**LSTM**). La versione attuale (V3.2) risolve le criticità dei modelli precedenti introducendo un'architettura a **Vettore Ibrido (Locale + Globale)** capace di distinguere micromovimenti (tremori) da macromovimenti (tic complessi/scatti del collo).

## 🚀 Key Features (V3.2)

* **🎥 Analisi Markerless "In the Wild":** Funziona su video standard con soggetti non vincolati, grazie a una robusta pipeline di **Normalizzazione Cinematica** (Ancoraggio al naso + Invarianza di scala).
* **🧠 Architettura Ibrida & "The Boost":** Combina 248 feature locali (occhi/bocca) con vettori di velocità globale della testa amplificati (Feature Scaling x100) per rilevare tic complessi e Head Jerks altrimenti invisibili.
* **⚡ Data Augmentation Fisiologica:** Training su dataset sintetici generati simulando le leggi fisiche dei sintomi (Oscillatori armonici per il tremore, funzioni Impulso per i Tic, Random Walk per la Discinesia).
* **📊 Reportistica Clinica Interattiva:** Genera automaticamente una dashboard HTML con statistiche, timeline degli eventi delle anomalie rilevate in loop per l'analisi medica.
* **🛡️ Stabilizzazione Temporale:** Algoritmi di post-processing (*Patience* & *Visual Hold*) per eliminare il flickering e fornire rilevazioni stabili e leggibili.

## 🩺 Patologie Supportate
Il modello classifica 6 stati distinti basati sulla cinematica del movimento:
1.  **🟢 NORMALE:** Movimento fisiologico.
2.  **🟠 TIC (Sindrome di Tourette):** Movimenti balistici, rapidi e scattosi (inclusi scatti del collo).
3.  **🔴 TREMORE (Parkinson/Essenziale):** Oscillazioni ritmiche
4.  **🟣 IPOMIMIA (Depressione/Parkinson):** Riduzione patologica dell'espressività e del gain motorio.
5.  **⚫ PARESI (Paralisi di Bell):** Asimmetria statica e dinamica (Droop).
6.  **🔵 DISCINESIA (Tardiva/Corea):** Movimenti involontari lenti, fluidi e caotici.

