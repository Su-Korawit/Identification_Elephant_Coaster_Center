# src/elephant_vision/train_reid.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from elephant_vision.reid_model import ElephantReIDModel
from elephant_vision.reid_dataset import get_reid_dataloader

def train_reid(data_dir="artifacts/datasets/reid/train", epochs=20, batch_size=16, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader, dataset = get_reid_dataloader(data_dir, batch_size=batch_size)
    num_classes = len(dataset.classes)
    print(f"📦 Found {num_classes} elephant identities: {dataset.classes}")

    model = ElephantReIDModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        acc = correct / total * 100
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # ✅ Save model
    os.makedirs("artifacts/models", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/models/reid_model.pt")
    print("✅ Model saved to artifacts/models/reid_model.pt")

    # ✅ Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss", color="royalblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["acc"], label="Training Accuracy", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.legend()

    os.makedirs("artifacts/plots", exist_ok=True)
    plt.tight_layout()
    plot_path = "artifacts/plots/reid_training_curves.png"
    plt.savefig(plot_path)
    print(f"📊 Saved training plot to {plot_path}")
    plt.show()

if __name__ == "__main__":
    train_reid()
