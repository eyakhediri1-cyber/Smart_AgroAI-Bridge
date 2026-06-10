"""
train_cnn.py — Train olive disease CNN on PlantVillage dataset
Usage: python train_cnn.py --data_dir ./plantvillage_olive --epochs 20
"""
import argparse, torch, torchvision
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

CNN_CLASSES = ["healthy", "peacock_eye", "anthracnose", "verticillium", "sooty_mold", "leaf_spot"]

def train(data_dir: str, epochs: int = 20, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # Aggressive augmentation to handle real-world photos vs PlantVillage lab shots
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_path = Path(data_dir)
    train_ds = datasets.ImageFolder(str(data_path / "train"), train_tf)
    val_ds   = datasets.ImageFolder(str(data_path / "val"),   val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"📁 Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"   Classes: {train_ds.classes}")

    # ResNet-18 transfer learning
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:02d}/{epochs} | loss={train_loss/len(train_dl):.4f} | val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "olive_cnn.pth")
            print(f"  💾 Saved best model (acc={acc:.3f})")

    print(f"\n✅ Training done. Best val acc: {best_acc:.3f}")
    print("   Model saved to: olive_cnn.pth")
    print("   Copy olive_cnn.pth to backend/olive_cnn.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./plantvillage_olive")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size)
