import torch

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            test_loss += loss_fn(preds, yb).item()
    return test_loss / len(test_loader)