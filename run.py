#!/usr/bin/env python3

import sys
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from net import Net

args = [arg.lower() for arg in sys.argv]
TRAIN = 'train' in args
TEST = 'test' in args

if not TRAIN and not TEST:
    print('Specify train or test')

transform = transforms.Compose([
    transforms.Resize((320, 180), Image.LANCZOS),
    transforms.ToTensor(),
])

data = torchvision.datasets.ImageFolder('pictures/', transform=transform)

size = len(data)
indices = list(range(size))
split = int(size * .8)
train_indices, test_indices = indices[:split], indices[split:]

train_data = torch.utils.data.Subset(data, train_indices)
test_data = torch.utils.data.Subset(data, test_indices)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, drop_last=True, num_workers=2)

classes = ('Yes', 'No')


net = Net()

if TEST and not TRAIN:
    net.load_state_dict(torch.load('./output_model.pth'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


if TRAIN:
    print('Beginning Training')

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss : {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

    torch.save(net.state_dict(), './output_model.pth')

if TEST:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')
