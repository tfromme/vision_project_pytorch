#!/usr/bin/env python3

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from net import Net

transform = transforms.Compose([
    transforms.ToTensor()
])

# train_loader = torch.utils.data.DataLoader(torchvision.datasets.VOCSegmentation('./voc/', download=True))

train_data = torchvision.datasets.ImageFolder('scaled_pictures/Train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

test_data = torchvision.datasets.ImageFolder('scaled_pictures/Test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

classes = ('Yes', 'No')


net = Net()
net.load_state_dict(torch.load('./output_model.pth'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
