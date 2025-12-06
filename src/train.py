import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Import dei nostri moduli
import config
from model import NeuroLSTM
from data_gen import genera_dataset_da_video

def train_model():
    print(f"--- AVVIO TRAINING NEUROMETRIC MODEL 2.0 ---")
    print(f"Device: {config.DEVICE}")
    print(f"Input Size: {config.INPUT_SIZE} features (coordinate x,y)")
    print(f"Classes: {config.NUM_CLASSES}")

    # 1. GENERAZIONE DATI (ETL)
    print("\n[FASE 1] Generazione Dataset Sintetico...")
    X_train, y_train = genera_dataset_da_video()

    if len(X_train) == 0:
        print("ERRORE: Nessun dato generato. Controlla i percorsi video.")
        return

    # Conversione in Tensor
    tensor_x = torch.Tensor(X_train) # Shape: (N, 30, 124)
    tensor_y = torch.LongTensor(y_train)

    # DataLoader (Batching)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    print(f"Dataset pronto: {len(dataset)} campioni.")
    print(f"Batch Size: {config.BATCH_SIZE} -> {len(dataloader)} iterazioni per epoca.")

    # 2. INIZIALIZZAZIONE MODELLO
    model = NeuroLSTM(
        input_size=config.INPUT_SIZE, 
        hidden_size=config.HIDDEN_SIZE, 
        num_layers=config.NUM_LAYERS, 
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)

    # Loss e Optimizer
    # CrossEntropyLoss include già LogSoftmax, ottimo per stabilità numerica
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Liste per i grafici
    loss_history = []
    acc_history = []
    best_acc = 0.0

    # 3. LOOP DI TRAINING
    print("\n[FASE 2] Inizio Addestramento...")
    model.train()

    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Reset gradienti
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()
            
            # Calcolo statistiche
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Statistiche Epoca
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        # Salva il modello se è il migliore finora
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            # print("  -> Nuovo record! Modello salvato.")

    print("\n[FASE 3] Training Completato.")
    print(f"Miglior Accuracy raggiunta: {best_acc:.2f}%")
    print(f"Modello salvato in: {config.MODEL_SAVE_PATH}")

    # 4. GENERAZIONE GRAFICI (REPORT)
    plt.figure(figsize=(12, 5))
    
    # Grafico Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss', color='red')
    plt.title('Andamento Loss (Errore)')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Grafico Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Training Accuracy', color='green')
    plt.title('Andamento Accuracy (Precisione)')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig('training_results.png')
    print("Grafico salvato come 'training_results.png'")

if __name__ == "__main__":
    train_model()