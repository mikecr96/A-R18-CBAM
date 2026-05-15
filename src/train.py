"""
Main training pipeline for the Asymmetric ResNet-18 + CBAM architecture.
Optimized for high-cardinality HDSR using Automatic Mixed Precision (AMP).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.asymmetric_resnet import AsymmResNet18CBAM
from utils.imageFolder import CustomImageFolder
from utils.randAugment import MyRandAugment

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Asymmetric ResNet-18 CBAM")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the PREP dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--save_dir', type=str, default='weights', help='Directory to save model checkpoints')
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """
    Executes a single training epoch with AMP hardware acceleration.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # AMP: Automatic Mixed Precision context
        # Usa float16/bfloat16 donde es seguro, ahorrando VRAM y acelerando los Tensor Cores
        with torch.autocast(device_type=device.type, enabled=device.type in ['cuda', 'mps']):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Escala la pérdida y hace el backward
        scaler.scale(loss).backward()
        
        # Unscales los gradientes y actualiza los pesos
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Loss': loss.item(), 'Acc': correct / total})
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.inference_mode()
def validate(model, dataloader, criterion, device, epoch):
    """
    Evaluates the model on the validation split.
    Using @torch.inference_mode() for maximum performance.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Validation")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Autocast también en inferencia para mayor velocidad
        with torch.autocast(device_type=device.type, enabled=device.type in ['cuda', 'mps']):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'Val Loss': loss.item(), 'Val Acc': correct / total})
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"🔥 Firing up the training pipeline on: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((64, 192)),
        MyRandAugment(num_ops=2, magnitude=6),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 192)),
        transforms.ToTensor(),
    ])

    print("📚 Loading dataset...")
    train_dataset = CustomImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = CustomImageFolder(root=os.path.join(args.data_dir, 'val'), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("🤖 Initializing Asymmetric ResNet-18 + CBAM...")
    model = AsymmResNet18CBAM(cbam_layers=[False, False, True, True], ratio=16, num_classes=1001).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Inicializando el GradScaler para AMP (Evita Underflow en float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        
        print(f"📈 Epoch {epoch} Summary | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "best_ar18_cbam_s34_r16.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 New SOTA! Model saved to {save_path}")

    print("✅ Training complete. ¡Malianteo científico puro!")

if __name__ == "__main__":
    main()