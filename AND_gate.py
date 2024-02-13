import torch
from torch import nn
import random


class NNSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam
        self.epochs = 2000
        self.learning_rate = 1e-1
        self.file_name = 'model_AND.pth'
        self.data_test = NNSigmoid.get_data()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_stack(x)

    @staticmethod
    def get_data():
        x = [[False, False],
             [False, True],
             [True, False],
             [True, True]]
        y = [False, False, False, True]
        data_test = []
        for i in range(len(x)):
            data_test.append((torch.FloatTensor(x[i]), torch.FloatTensor([y[i]])))
        return data_test

    @staticmethod
    def get_shuffle_set(size):
        initial = []
        for i in range(size):
            initial.append(i)
        shuffle_set = []
        while len(initial) > 1:
            seq = random.randint(0, len(initial) - 1)
            shuffle_set.append(initial.pop(seq))
        shuffle_set.append(initial.pop(0))
        return shuffle_set

    def train_me(self):
        self.train()
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            for i in self.get_shuffle_set(len(self.data_test)):
                X, y = self.data_test[i]

                # Compute prediction error
                pred = self(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 200 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f}  [{epoch}/{self.epochs}]")

    def test(self):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        self.eval()
        test_loss = 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in self.data_test:
                pred = self(X)
                print(f" The goal / prediction: {y} / {pred}")
                test_loss += self.loss_fn(pred, y).item()
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    def get_model(self):
        self.train_me()
        self.test()
        torch.save(self.state_dict(), self.file_name)
        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def test_model(self):
        self.load_state_dict(torch.load(self.file_name))
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        self.test()


if __name__ == '__main__':
    model = NNSigmoid()
    model.get_model()
    model.test_model()



