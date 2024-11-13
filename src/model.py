import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from typing import List

'''
    @author Dmytro Gnatyk
    @param This is a simple CNN model using PyTorch
'''

# Device
device: List[str] = ['cuda:0' if torch.cuda.is_available() else 'cpu']

# Transform
transform: transforms.Compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_set: MNIST = MNIST(root='./data', train=True, download=True, transform=transform)
data_loader: DataLoader = DataLoader(data_set, batch_size=32, shuffle=True)

for images, label in data_loader:
    print(f"Image batch size shape -> {images.shape}")
    print(f"Image label shape -> {label.shape}")
    break

# Classes of Images
classes: List[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.conv_layer: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer: nn.Sequential = nn.Sequential(
            nn.Linear(128 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

model: CNN = CNN().to(device[0])
print(model)

# Loss and Optimizer
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: Adam = Adam(model.parameters(), lr=0.001)

# Training
num_epochs: int = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device[0])
        labels = labels.to(device[0])

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Step: {i + 1}, Loss: {loss.item()}")

def main() -> None:
    print("Saving Model...")
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()
