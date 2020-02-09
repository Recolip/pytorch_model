from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms,datasets
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import time
import math
import numpy as np
import os,cv2
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import torch.nn.init as init


batch_size = 4
num_classes = 4
learning_rate = 0.001
momentum = 0.9
img_size = 224
# 定义一个类，需要创建模型的时候，就实例化一个对象

class VehicleColorRecognitionModel(nn.Module):
    def __init__(self,Load_VIS_URL=None):
        super(VehicleColorRecognitionModel,self).__init__()
        
        # ===============================  top ================================
        # first top convolution layer   
        self.top_conv1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(3, 48, kernel_size=(11,11), stride=(4,4)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
       
        
        # first top convolution layer    after split
        self.top_top_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.top_bot_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        
        #  need a concat
        
        # after concat  
        self.top_conv3 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(128, 192, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        # fourth top convolution layer
        # split feature map by half
        self.top_top_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        self.top_bot_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        
        # fifth top convolution layer
        self.top_top_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.top_bot_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

#        # ===============================  bottom ================================
    
           
#         # first bottom convolution layer   
        self.bottom_conv1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(3, 48, kernel_size=(11,11), stride=(4,4)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
       
        
        # first top convolution layer    after split
        self.bottom_top_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.bottom_bot_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        
        #  need a concat
        
        # after concat  
        self.bottom_conv3 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(128, 192, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        # fourth top convolution layer
        # split feature map by half
        self.bottom_top_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        self.bottom_bot_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        
        # fifth top convolution layer
        self.bottom_top_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.bottom_bot_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Fully-connected layer
        self.classifier = nn.Sequential(
            nn.Linear(5*5*64*4, 4096), 
            nn.ReLU(), 
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.ReLU(), 
            nn.Dropout(0.6),
            nn.Linear(4096, num_classes)
        )                
        
    def forward(self,x):
        #print(x.shape)
        x_top = self.top_conv1(x)
        #print(x_top.shape)
                
        x_top_conv = torch.split(x_top, 24, 1)
        
        x_top_top_conv2 = self.top_top_conv2(x_top_conv[0])
        x_top_bot_conv2 = self.top_bot_conv2(x_top_conv[1])
        
        x_top_cat1 = torch.cat([x_top_top_conv2,x_top_bot_conv2],1)
        
        x_top_conv3 = self.top_conv3(x_top_cat1)
        
        x_top_conv3 = torch.split(x_top_conv3, 96, 1)
        
        x_top_top_conv4 = self.top_top_conv4(x_top_conv3[0])
        x_top_bot_conv4 = self.top_bot_conv4(x_top_conv3[1])
        
        x_top_top_conv5 = self.top_top_conv5(x_top_top_conv4)
        x_top_bot_conv5 = self.top_bot_conv5(x_top_bot_conv4)
        
        x_bottom = self.bottom_conv1(x)
        
        x_bottom_conv = torch.split(x_bottom, 24, 1)
        
        x_bottom_top_conv2 = self.bottom_top_conv2(x_bottom_conv[0])
        x_bottom_bot_conv2 = self.bottom_bot_conv2(x_bottom_conv[1])
        
        x_bottom_cat1 = torch.cat([x_bottom_top_conv2,x_bottom_bot_conv2],1)
        
        x_bottom_conv3 = self.bottom_conv3(x_bottom_cat1)
        
        x_bottom_conv3 = torch.split(x_bottom_conv3, 96, 1)
        
        x_bottom_top_conv4 = self.bottom_top_conv4(x_bottom_conv3[0])
        x_bottom_bot_conv4 = self.bottom_bot_conv4(x_bottom_conv3[1])
        
        x_bottom_top_conv5 = self.bottom_top_conv5(x_bottom_top_conv4)
        x_bottom_bot_conv5 = self.bottom_bot_conv5(x_bottom_bot_conv4)
        
        x_cat = torch.cat([x_top_top_conv5,x_top_bot_conv5,x_bottom_top_conv5,x_bottom_bot_conv5],1)
        
        
        flatten = x_cat.view(x_cat.size(0), -1)
        
        output = self.classifier(flatten)
        
        #output = F.softmax(output)
        
        
        return output
if __name__ == "__main__":

    classes = ['images_#303765', 'images_#97264f', 'images_#aa4734', 'images_#ac3122']

    # Load Model and 1 forward passs
    load_model = VehicleColorRecognitionModel()
    #load_model.load_state_dict(torch.load("./trained_parameter/Gloriot_lipstick_net.pth"))
    load_model.load_state_dict(torch.load("./lipstick_net_test1.pth"))

    # Load Data from image folder
    input_path = "../uploader/public/inputs/"
    
    # Apply image transfomation
    transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.CenterCrop(224),
        transforms.ToTensor()
     ],
    )

    input_dataset = torchvision.datasets.ImageFolder(
        root=input_path,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        input_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    with torch.no_grad():
        iter_ = iter(loader)
        img_tensor = iter_.next()[0]
        outputs = load_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        print(classes[predicted])
    import json

    data = {}
    data['lipstick'] = []
    print (predicted)
    url = 'https://www.maybelline.com/lip-makeup/lipstick/superstay-matte-ink-city-edition-liquid-lipstick-makeup'
    lt_webhook = 'https://nice-dolphin-18.localtunnel.me/lipsticksimg'
    if(predicted == 0):
        data['lipstick'].append({
            'color': '#303765',
            'name': 'Explorer',
            'image': lt_webhook+'/explorer.png',
            'url': url
        })
        
    elif(predicted == 1):
        data['lipstick'].append({
        'color': '#97264F',
        'name': 'Artist',
        'image': lt_webhook+'/artist.png',
        'url': url
        })
    elif(predicted == 2):
        data['lipstick'].append({
        'color': '#AA4734',
        'name': 'Globetrotter',
        'image': lt_webhook+'/globe.png',
        'url': url
        })
    elif(predicted == 3):
        data['lipstick'].append({
        'color': '#AC3122',
        'name': 'Dancer',
        'image': lt_webhook+'/dancer.png',
        'url': url
        })
        
    with open('../output/output.json', 'w') as outfile:
        json.dump(data, outfile)    

    import requests
    r = requests.post("https://heavy-eel-2.localtunnel.me/result", json=data)
    print(r.status_code)
    print(r)
