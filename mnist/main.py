import time

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(784, 128)
        self.output_layer = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.loss_history = []

    def forward(self, x):
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def fit(self, X, y, epochs=10, batch_size=256, lr=0.005):
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_inputs, batch_targets in data_loader:
                outputs = self(batch_inputs)
                loss = loss_fn(outputs, batch_targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_inputs.size(0)

            epoch_loss /= len(dataset)
            self.loss_history.append(epoch_loss)
            print(f'Epoch {epoch}, Loss {epoch_loss:.4f}')

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            predicted_classes = torch.argmax(outputs, dim=1)
        return predicted_classes

def _load_mnist_training_data():
    mnist_train = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

    X_train = mnist_train.data.float() / 255.0
    y_train = mnist_train.targets
    X_train = X_train.view(X_train.size(0), -1)

    return X_train, y_train

def _load_mnist_testing_data():
    mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

    X_test = mnist_test.data.float() / 255.0
    y_test = mnist_test.targets
    X_test = X_test.view(X_test.size(0), -1)

    return X_test, y_test


def _visualize(model, predicted_classes, true_classes, X_test):
    save_dir = Path(__file__).parent

    #Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig(save_dir / "mnist_loss_curve.png")
    plt.close()

    num_samples = 10
    indices = torch.randperm(X_test.size(0))[:num_samples]

    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {true_classes[idx]}\nPred: {predicted_classes[idx]}")
        plt.axis('off')
    plt.suptitle("Sample MNIST Predictions")
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_sample_predictions.png")
    plt.close()

def main():
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "mnist_model.pth"

    if model_path.exists():
        model = NeuralNetwork()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        start_time = time.time()
        torch.manual_seed(20)
        X_train, y_train = _load_mnist_training_data()
        model = NeuralNetwork()
        model.fit(X_train, y_train)
        torch.save(model.state_dict(), model_path)
        end_time = time.time()
        print(f"Model trained and saved in: {end_time - start_time:.6f} seconds")

    X_test, y_test = _load_mnist_testing_data()
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    _visualize(model, predictions, y_test, X_test)

if __name__ == "__main__":
    main()