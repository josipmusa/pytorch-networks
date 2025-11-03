from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def _visualize(model, inputs, targets, loss_history):
    save_dir = Path(__file__).parent
    #Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("XOR Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "xor_loss_curve.png", dpi=150)
    plt.close()

    #Decision boundary
    x_min, x_max = -0.1, 1.1
    y_min, y_max = -0.1, 1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid)
        preds = preds.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds, levels=50, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Output Probability")

    # Plot training points
    for i, label in enumerate(targets.flatten()):
        color = "white" if label == 0 else "black"
        plt.scatter(inputs[i, 0], inputs[i, 1], c=color, edgecolors="k", s=100)

    plt.title("XOR Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.tight_layout()
    plt.savefig(save_dir / "xor_decision_boundary.png", dpi=150)
    plt.close()

def main():
    torch.manual_seed(20)
    model = NeuralNetwork()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    inputs = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])

    targets = torch.tensor([
        [0.],
        [1.],
        [1.],
        [0.]
    ])

    loss_history = []
    for epoch in range(20000):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    outputs = model(inputs).detach()
    predictions = (outputs > 0.5).float()

    print("\nXOR Network Predictions:")
    print("Input\t\tRaw Output\tPredicted\tExpected")
    for i in range(len(inputs)):
        input_vals = inputs[i].tolist()
        raw_output = outputs[i].item()
        predicted = int(predictions[i].item())
        expected = int(targets[i].item())
        print(f"{input_vals}\t{raw_output:.4f}\t\t{predicted}\t\t{expected}")

    _visualize(model, inputs, targets, loss_history)

if __name__ == '__main__':
    main()