import torch
from torch import nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


def main():
    model = NeuralNetwork()

    optimizer = optim.SGD(model.parameters(), lr=0.5)
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

    for epoch in range(10000):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

if __name__ == '__main__':
    main()