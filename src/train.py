import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import config
from model import NeuroLSTM
from data_gen import genera_dataset_da_video

def train():
    print("--- 1. GENERAZIONE DATI ---")
    X_np, y_np = genera_dataset_da_video()
    
    if len(X_np) == 0:
        print("❌ ERRORE: Dataset vuoto. Controlla i video.")
        return

    # Converti in Tensori PyTorch
    tensor_x = torch.from_numpy(X_np)
    tensor_y = torch.from_numpy(y_np)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    print("--- 2. INIZIALIZZAZIONE MODELLO ---")
    model = NeuroLSTM(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"--- 3. AVVIO TRAINING SU {config.DEVICE} ---")
    model.train()
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if (epoch+1) % 5 == 0:
            acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")
            
    # Salvataggio
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\n✅ MODELLO SALVATO: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()