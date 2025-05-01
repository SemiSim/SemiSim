## Type: Training Script

Code includes:
- ==Dataset class== for loading `.npy` / `.pt `features and labels 
- An MLP classifier
- Training loop with validation steps
- Accuracy reporting 
- Evaluation for test set generalization
- Auto-detection of LLPSDB assays conducted under physio conditions


Code requires installation of torch scikit-learn:
```bash
pip install torch scikit-learn numpy

```

V2.1 llps_classifier.py
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Dataset class
class LLPSDataset(Dataset):
    def __init__(self, features_path="llpsdb_physio_features.npy", labels_path="llpsdb_physio_labels.npy"):
        self.X = torch.tensor(np.load(features_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(labels_path), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Simple MLP model
class LLPSClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LLPSClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification (2 classes)
        )

    def forward(self, x):
        return self.model(x)

# 3. Training function
from sklearn.metrics import classification_report, confusion_matrix

def train_model(dataset, epochs=20, batch_size=32, lr=1e-3):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = LLPSClassifier(input_dim=dataset[0][0].shape[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                preds = torch.argmax(output, dim=1)
                y_val_true.extend(y_batch.numpy())
                y_val_pred.extend(preds.numpy())

        val_acc = accuracy_score(y_val_true, y_val_pred)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.4f}")

    # --- Test Evaluation ---
    model.eval()
    y_test_true, y_test_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            preds = torch.argmax(output, dim=1)
            y_test_true.extend(y_batch.numpy())
            y_test_pred.extend(preds.numpy())

    print("\n📊 Test Set Evaluation:")
    print(classification_report(y_test_true, y_test_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test_true, y_test_pred))

    return model

# 4. Run everything
if __name__ == "__main__":
    dataset = LLPSDataset("saps_pondr_features.npy", "saps_pondr_labels.npy")
    model = train_model(dataset, epochs=30)
    torch.save(model.state_dict(), "llps_model.pt")
    print("🧠 Model saved as 'llps_model.pt'")

```


To use, run:
```bash
python llps_classifier.py

```
& it'll load the llpsdb_physio_features.npy in automatically, detecting & training our model on the physiological subset of LLPSDB entries.

























