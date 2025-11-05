"""
Step 3: Train LSTM model for frequency filtering
Framework: PyTorch

PyTorch is chosen for:
1. Excellent LSTM implementation with flexible architecture
2. Dynamic computation graphs (easier debugging)
3. Strong community support and documentation
4. Industry standard for research and production
5. Better control over training process
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Load Training Data
# ============================================================================
print("\nLoading training data...")
data = np.load('data/training_data.npz')
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']
sequence_length = int(data['sequence_length'])

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# ============================================================================
# Hyperparameters (Based on research for time series LSTM)
# ============================================================================
print("\n" + "="*70)
print("HYPERPARAMETERS")
print("="*70)

# Model hyperparameters
input_size = 5  # 1 signal + 4 selector values
hidden_size = 128  # Number of LSTM units (64-256 typical for this task)
num_layers = 2  # Number of stacked LSTM layers (2-3 common)
output_size = 1  # Single frequency value
dropout = 0.2  # Dropout rate (0.2-0.3 helps prevent overfitting)

# Training hyperparameters
batch_size = 64  # Batch size (32-128 typical)
learning_rate = 0.001  # Adam default, works well for LSTMs
num_epochs = 50  # Training epochs
weight_decay = 1e-5  # L2 regularization

print(f"""
Model Architecture:
  - Input size: {input_size} (signal + selector)
  - Hidden size: {hidden_size} LSTM units
  - Number of layers: {num_layers} stacked LSTMs
  - Output size: {output_size}
  - Dropout: {dropout}

Training Configuration:
  - Batch size: {batch_size}
  - Learning rate: {learning_rate}
  - Optimizer: Adam with weight decay {weight_decay}
  - Loss function: MSE (Mean Squared Error)
  - Epochs: {num_epochs}
  - Device: {device}

Rationale:
  - Hidden size 128: Sufficient capacity for 4 frequency components
  - 2 layers: Captures temporal patterns at different scales
  - Dropout 0.2: Prevents overfitting while maintaining performance
  - Adam optimizer: Adaptive learning rate, robust for LSTMs
  - MSE loss: Standard for regression, penalizes amplitude errors
""")

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================
# Define LSTM Model
# ============================================================================
class FrequencyFilterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FrequencyFilterLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=False  # Unidirectional for causal processing
        )
        
        # Dropout layer after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to map LSTM output to frequency value
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        
        # LSTM forward pass
        # lstm_out shape: (batch, sequence_length, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply fully connected layer to each time step
        # output shape: (batch, sequence_length, output_size)
        output = self.fc(lstm_out)
        
        return output

# ============================================================================
# Initialize Model
# ============================================================================
model = FrequencyFilterLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    dropout=dropout
).to(device)

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# Loss Function and Optimizer
# ============================================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler (reduce on plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients in LSTMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# ============================================================================
# Training Loop
# ============================================================================
print("\n" + "="*70)
print("TRAINING")
print("="*70)

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 15

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    
    # Record losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    epoch_time = time.time() - epoch_start
    
    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Time: {epoch_time:.2f}s")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.2f}s")
print(f"Best validation loss: {best_val_loss:.6f}")

# ============================================================================
# Save Training History
# ============================================================================
Path("models").mkdir(exist_ok=True)
np.savez('models/training_history.npz',
         train_losses=train_losses,
         val_losses=val_losses,
         num_epochs=len(train_losses))

# ============================================================================
# Visualize Training Progress
# ============================================================================
Path("visualizations").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
epochs_range = range(1, len(train_losses) + 1)
ax.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
ax.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Validation Loss')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Training and Validation Loss Over Time', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # Log scale for better visualization
plt.tight_layout()
plt.savefig('visualizations/07_training_loss.png', dpi=300, bbox_inches='tight')
print("\nSaved: visualizations/07_training_loss.png")
plt.close()

# ============================================================================
# Load Best Model for Evaluation
# ============================================================================
model.load_state_dict(torch.load('models/best_model.pth'))
print("\nLoaded best model for evaluation")

print("\nTraining script completed!")
