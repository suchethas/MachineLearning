from __future__ import print_function, division
import numpy
import pandas as pd
import xlwt
from xlwt import Workbook
from data_loader import ImageFolder
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torchvision.utils as utils
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import csv
#FOLDER_DATASET = "/Users/suchethas/Documents/cifar100"
plt.ion()

class DriveData(Dataset):
    __xs = []
    __ys = []
    #__path = []
    
    def __init__(self, folder_dataset, transform):
        self.transform = transform
        with open("da.csv", 'r+') as csvfile:
            read = csv.reader(csvfile, delimiter=',')
            for row in read:
                self.__xs.append(row[0])       
                # Steering wheel label
                self.__ys.append(row[1])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        #print(self.__xs[index])
        img1 = Image.open(self.__xs[index])
        img1 = img1.convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
        # Convert image and label to torch tensors
        img1 = torch.from_numpy(np.asarray(img1))
        
        img2 = Image.open(self.__ys[index])
        img2 = img2.convert('RGB')
        if self.transform is not None:
            img2 = self.transform(img2)
        # Convert image and label to torch tensors
        img2 = torch.from_numpy(np.asarray(img2))
        
        #self.__path.append(self.__xs[index])
        return img1, img2, self.__xs[index]#self.__path

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)


transform = transforms.Compose(
    [transforms.ToTensor(),])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_dir = '/Users/suchethas/Documents/cifar100'
res_dir = 'results_autoencoder_cifar100'
image = DriveData(data_dir, transform=transform)
dataloaders = torch.utils.data.DataLoader(image, batch_size = 200, shuffle = True, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
        
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), # input_channels, output_channels, filter_size, stride
            nn.ReLU(True),
            nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True), #inplace = True
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),

        )

    def forward(self,x):
        x=self.encoder(x) # x.shape = torch.Size([1, 5, 14, 14])
        x=self.decoder(x)
        return x

    
model = Autoencoder()
model.to(device)
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 20
torch.save(model.state_dict(),'cifar100_auto_model.pth')
#def imshow_decoder(ima,p):
    #with torch.no_grad():
        #ima[0] = (ima[0] - torch.min(ima[0]))/(torch.max(ima[0]) - torch.min(ima[0]))
        #save_path = os.path.join(res_dir, p.split('/')[5],p.split('/')[6].split(".")[0])
        #create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
        #utils.save_image(output[xi,:,:,:], save_path + '_auto.png',  normalize=True, padding=0)

#res_dir = 'cifar100_results'
model.train()
for epoch in range(num_epochs):
    for batch, inp in enumerate(dataloaders,1):
        img = inp[0]
        tar = inp[1]
        paths = inp[2]
        #print(paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0])
#         print(os.path.join(res_dir, paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0]))
#         print(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
        ad=img.to(device)
        # ad.requiers_grad = True
        ad_tar=tar.to(device)
        # ad_tar.requiers_grad = True
        

        # ===================forward=====================
        output = model(ad)
        
        if epoch == num_epochs-1:
            with torch.no_grad():
                for xi in range(200):
                    #imshow_decoder(output[xi], paths[xi])
                    save_path = os.path.join(res_dir, paths[xi].split('/')[5],paths[xi].split('/')[6].split(".")[0])
                    create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
                    #ima[0] = (ima[0] - torch.min(ima[0]))/(torch.max(ima[0]) - torch.min(ima[0]))
                    utils.save_image(output[xi,:,:,:], save_path + '_auto.png',  normalize=True, padding=0)
	
        loss = distance(output, ad_tar)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
