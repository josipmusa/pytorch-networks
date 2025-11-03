import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_history = []

    def forward(self, x):
        x = self.sigmoid(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x

    def fit(self, X, y, epochs=10000, batch_size=64, lr=0.2):
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_inputs, batch_targets in data_loader:
                outputs = self(batch_inputs)
                loss = loss_fn(outputs, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_inputs.size(0)

            epoch_loss /= len(dataset)
            self.loss_history.append(epoch_loss)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss {epoch_loss:.4f}')

    def predict(self, X):
        with torch.no_grad(): #Don't compute gradients
            outputs = self(X)
            predicted_classes = (outputs.flatten() > 0.5).int()
        return predicted_classes

def _create_train_data():
    X = np.random.rand(200, 2)
    y = (X[:, 1] > 0.5 * X[:, 0] + 0.2).astype(int).reshape(-1, 1)  # reshape so it matches activations

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

def _create_test_data():
    # points below the line → expected class 0
    test_points_0 = np.array([
        [0.1, 0.1],
        [0.3, 0.2],
        [0.5, 0.3]
    ])

    # points above the line → expected class 1
    test_points_1 = np.array([
        [0.1, 0.5],
        [0.4, 0.6],
        [0.6, 0.7]
    ])

    X_test = np.vstack((test_points_0, test_points_1))
    y_test = (X_test[:, 1] > 0.5 * X_test[:, 0] + 0.2).astype(int)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_test_tensor, y_test_tensor

def _visualize(model, X_train, y_train):
    save_dir = Path(__file__).parent

    #Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(model.loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Binary Classification Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "binary_classification_loss_curve.png")
    plt.close()

    #Decision boundary
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        zz = model(grid_tensor).numpy().reshape(xx.shape)
        predicted = (zz > 0.5).astype(int)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, predicted, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), cmap="coolwarm", edgecolors='k')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.tight_layout()
    plt.savefig(save_dir / "binary_classification_decision_boundary.png")
    plt.close()


def main():
    torch.manual_seed(20)
    X_train, y_train = _create_train_data()

    model = NeuralNetwork()
    model.fit(X_train, y_train)

    X_test, y_test = _create_test_data()
    predicted_classes = model.predict(X_test)

    print(" x1     x2   | Expected  Predicted")
    print("---------------------------------")
    for i in range(len(X_test)):
        print(f"{X_test[i, 0]:.2f}  {X_test[i, 1]:.2f}  |    {y_test[i]}        {predicted_classes[i]}")

    _visualize(model, X_train, y_train)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.6f} seconds")