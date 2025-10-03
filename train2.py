import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, vgg11 , vgg16_bn, VGG16_BN_Weights, VGG11_Weights
from torch.utils.data import DataLoader, random_split
import torch
torch.cuda.empty_cache()
from outils import set_seed, evaluate
from transformation import GetTransforms  

def train_model(lr, batch_size, num_epochs, transform_factory, val_frac=0.1, early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build transforms.
    train_tf = transforms.Compose(transform_factory.trainparams())
    test_tf  = transforms.Compose(transform_factory.testparams())
    # 2) Download CIFAR‑10 train / test
    full_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )

    # 3) Split train → train/val
    n_val   = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val]
        #generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # 4) DataLoaders
    train_loader = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,    batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # 5) Build & adapt VGG16
    #model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)  
    model = vgg16(weights=None)

    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, 10) # type: ignore
    model = model.to(device)

    # 6) Optimizer & loss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_no_improve = 0
    # 7) Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        running_correct = 0
        running_total   = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total   += y.size(0)

        train_acc = running_correct / running_total
        val_acc   = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{num_epochs}  –  "
              f"train acc: {train_acc:.4f}  –  val acc: {val_acc:.4f}")
        """
    # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break
        """
    # 8) Final test accuracy
    test_acc = evaluate(model, test_loader, device)
    print(f"Final test acc: {test_acc:.4f}")

    # 9) Return state dict 
    return model.state_dict()
    
tf = GetTransforms()
"""
state_dict1 = train_model(
    lr=0.01,
    batch_size=64,
    num_epochs=70,
    transform_factory=tf,
    val_frac=0.1,
    early_stopping_patience=5
)
torch.save(state_dict1, "vgg16_cifar10_7.pth")
"""
state_dict2 = train_model(
    lr=0.01,
    batch_size=64,
    num_epochs=70,
    transform_factory=tf,
    val_frac=0.1,
    early_stopping_patience=5
)
torch.save(state_dict2, "vgg16_cifar10_8.pth")

"""
import os

def continue_training(weights_path, lr, batch_size, num_epochs, transform_factory, 
                     val_frac=0.1, early_stopping_patience=5, start_epoch=25):
    ""
    Continue training a VGG11 model from saved weights
    
    Args:
        weights_path: Path to saved model weights (.pth file)
        lr: Learning rate
        batch_size: Batch size for training
        num_epochs: Additional epochs to train
        transform_factory: Transform factory for data augmentation
        val_frac: Fraction of training data to use for validation
        early_stopping_patience: Epochs to wait before early stopping
        start_epoch: Starting epoch number (for display purposes)
    ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1) Build transforms
    train_tf = transforms.Compose(transform_factory.trainparams())
    test_tf = transforms.Compose(transform_factory.testparams())
    
    # 2) Download CIFAR-10 train / test
    full_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    test_ds = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )
    
    # 3) Split train → train/val
    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    # 4) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 5) Build VGG11 model (same architecture as before)
    model = vgg11(weights=None)
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, 10) # type: ignore
    
    # 6) Load saved weights
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("Weights loaded successfully!")
    else:
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    
    model = model.to(device)
    
    # 7) Evaluate current model performance before continuing training
    initial_val_acc = evaluate(model, val_loader, device)
    initial_test_acc = evaluate(model, test_loader, device)
    print(f"Initial performance - Val acc: {initial_val_acc:.4f}, Test acc: {initial_test_acc:.4f}")
    
    # 8) Optimizer & loss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = initial_val_acc
    epochs_no_improve = 0
    
    # 9) Continue training loop
    print(f"\nContinuing training for {num_epochs} more epochs...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)
            running_loss += loss.item()
            
            
        
        train_acc = running_correct / running_total
        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{start_epoch + num_epochs - 1}  –  "
              f"train acc: {train_acc:.4f}  –  val acc: {val_acc:.4f}  –  "
              f"avg loss: {avg_loss:.4f}")
        ""
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), weights_path.replace('.pth', '_best.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break
        ""
    # 10) Final test accuracy
    final_test_acc = evaluate(model, test_loader, device)
    print(f"Final test acc: {final_test_acc:.4f}")
    print(f"Best validation acc: {best_val_acc:.4f}")

    
    return model.state_dict()


state_dict1 = continue_training(
    weights_path="vgg11_cifar10_10.pth",
    lr=0.01,
    batch_size=64,
    num_epochs=15,
    transform_factory=tf,
    val_frac=0.1,
    early_stopping_patience=5,
    start_epoch=51
)
torch.save(state_dict1, "vgg11_cifar10_12.pth")
state_dict2 = continue_training(
    weights_path="vgg11_cifar10_11.pth",
    lr=0.01,
    batch_size=64,
    num_epochs=15,
    transform_factory=tf,
    val_frac=0.1,
    early_stopping_patience=5,
    start_epoch=51
)
torch.save(state_dict2, "vgg11_cifar10_13.pth")
"""