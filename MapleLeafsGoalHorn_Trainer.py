#!/usr/bin/python
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets, models, transforms
import torch.nn as nn
import tqdm
#### WHEN TESTING, USE THE SIGMOID ACTIVATION FUNCTION TO DETERMINE IF IT IS ONE
#### OR A ZERO
class LEAFSGOAL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(200704,1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, xb):
        return self.network(xb)
if True:
    model = torch.load("MapleLeafsGoalDetector.pt")
else:
    model = models.resnet18(pretrained=True)
    for params in model.parameters():
      params.requires_grad_ = False

    nr_filters = model.fc.in_features  #number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)
    print(nr_filters)
transform = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
                                           )
                                ])

dataset = datasets.ImageFolder('../Train/', transform=transform)
v_data = datasets.ImageFolder('../Test/', transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(v_data, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss = torch.nn.modules.loss.BCEWithLogitsLoss()
print(len(trainloader))
print(len(testloader))

averr = []

epoch_train_losses = []
epoch_test_losses = []
val_losses = []

for epoch in range(20):
    print(epoch+1)
    epoch_loss = 0
    for x_batch, y_batch in trainloader: #iterate ove batches
        y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape

        #make prediction
        yhat = model(x_batch)
        #enter train mode
        model.train()
        #compute loss
        l = loss(yhat,y_batch)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        #optimizer.cleargrads()
        epoch_loss += l/len(trainloader)

    epoch_train_losses.append(epoch_loss.detach().numpy())
    print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

    #validation doesnt requires gradient
    with torch.no_grad():
        cum_loss = 0
        for x_batch, y_batch in testloader:
            y_batch = y_batch.unsqueeze(1).float()

            #model to eval mode
            model.eval()

            yhat = model(x_batch)
            val_loss = loss(yhat,y_batch)
            cum_loss += val_loss/len(testloader)
            val_losses.append(val_loss.item())
    epoch_test_losses.append(cum_loss.detach().numpy())

torch.save(model, "MapleLeafsGoalDetector.pt")

plt.figure()

plt.ylim([0, 1])
plt.plot(epoch_test_losses)
plt.plot(epoch_train_losses)
plt.show()
