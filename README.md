
# Comparing Single-Layer RNN vs. Stacked RNN

## Overview

This project compares the performance of a Single-Layer RNN and a Stacked (Deep) RNN on a synthetic sequential dataset. The models are trained to predict the mean of a given input sequence, and their performance is analyzed by comparing loss trends during training.

## Features

1. Implements Single-Layer RNN using PyTorch.

2. Implements Stacked RNN with multiple layers.

3. Generates synthetic sequential data for training.

4. Compares training losses for both architectures.

5. Visualizes loss trends and model predictions.

### Requirements

Ensure you have the following dependencies installed:



# Comparing Single-Layer RNN vs. Stacked RNN

## Overview

This project compares the performance of a Single-Layer RNN and a Stacked (Deep) RNN on a synthetic sequential dataset. The models are trained to predict the mean of a given input sequence, and their performance is analyzed by comparing loss trends during training.

## Features

1. Implements Single-Layer RNN using PyTorch.

2. Implements Stacked RNN with multiple layers.

3. Generates synthetic sequential data for training.

4. Compares training losses for both architectures.

5. Visualizes loss trends and model predictions.

### Requirements

Ensure you have the following dependencies installed:


pip install torch numpy matplotlib
# Comparing Single-Layer RNN vs. Stacked RNN

## Overview

This project compares the performance of a Single-Layer RNN and a Stacked (Deep) RNN on a synthetic sequential dataset. The models are trained to predict the mean of a given input sequence, and their performance is analyzed by comparing loss trends during training.

## Features

1. Implements Single-Layer RNN using PyTorch.

2. Implements Stacked RNN with multiple layers.

3. Generates synthetic sequential data for training.

4. Compares training losses for both architectures.

5. Visualizes loss trends and model predictions.

### Requirements

Ensure you have the following dependencies installed:


```bash
pip install torch numpy matplotlib
```

## Implementation

### 1. Generate Data

Synthetic sequential data is generated where the target output is the mean of the input sequence.

```bash
def generate_data(seq_length=20, num_samples=1000):
    X = np.random.rand(num_samples, seq_length, 1)  # Random sequences
    y = X.mean(axis=1)  # Target: mean of sequence
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

### 2. Define Models

#### Single-Layer RNN

```bash
class SingleLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Using the last time-step output
        return out
```

#### Stacked RNN

```bash
class StackedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(StackedRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Using the last time-step output
        return out
```

### 3. Train and Compare Models

Models are trained using Mean Squared Error (MSE) loss and Adam optimizer.


```bash
def train(model, X_train, y_train, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses
```

### 4. Visualization

Loss trends for both models are plotted to compare their training performance.

```bash
plt.figure(figsize=(8,5))
plt.plot(single_rnn_losses, label='Single-Layer RNN', color='red')
plt.plot(stacked_rnn_losses, label='Stacked RNN', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Comparison: Single vs Stacked RNN')
plt.legend()
plt.show()
```

## Results

Stacked RNN tends to achieve lower loss than the Single-Layer RNN, indicating better learning capacity.

Training loss curves show how deeper RNNs capture complex patterns more effectively.

## Author

Developed by Pranit Chute

## License

This project is open-source and available under the MIT License.