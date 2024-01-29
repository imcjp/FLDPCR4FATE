import torch.nn as nn
from torchvision import transforms
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
# McMahan et al., 2016; 1,663,370 parameters
class CNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, num_hiddens=32, num_classes=10):
        super(CNN, self).__init__()
        self.trans=transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        self.activation = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5),
                               padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2) * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=False)

    def forward(self, x):

        # x = x.view(-1, 1, 28, 28)
        x = self.trans(x)
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x