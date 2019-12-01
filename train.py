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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
