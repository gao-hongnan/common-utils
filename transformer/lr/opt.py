import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create synthetic dataset
torch.manual_seed(42)
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X + torch.randn(X.size()) * 0.3

# Define simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Define training function with interventions
def train_model(intervention='baseline'):
    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1)  # High learning rate to create divergence
    losses = []

    for epoch in range(1000):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss_value = loss.item()
        losses.append(loss_value)

        # Check for divergence
        if loss_value > 1e3 and epoch > 50:
            if intervention == 'reset_optimizer':
                optimizer = optim.SGD(model.parameters(), lr=1)
            elif intervention == 'reduce_lr':
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            elif intervention == 'combination':
                optimizer = optim.SGD(model.parameters(), lr=0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

# Run experiments
strategies = ['baseline', 'reset_optimizer', 'reduce_lr', 'combination']
loss_curves = {}

for strategy in strategies:
    losses = train_model(intervention=strategy)
    loss_curves[strategy] = losses

# Visualize
plt.figure(figsize=(10, 6))
for strategy, losses in loss_curves.items():
    plt.plot(losses, label=strategy)
plt.yscale('log')  # Log scale to handle large loss values
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
