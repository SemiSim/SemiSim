Evaluating for model generalization

# Splits dataset into 3 subsets for evaluation:
1. Training Data 
2. Validation Data 
3. Test Data


- [ ] Replace `train_model()` in simpysim_v1.py with the following block of code: 

```python
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

```














































