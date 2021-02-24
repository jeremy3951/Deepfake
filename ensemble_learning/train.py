
from data import maskdata , deepfakedata
from Xcept import  xception , Xception 
import time
import torch
from torchvision import datasets, models, transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader ,Dataset , random_split

import numpy as np
import joblib
from efficientnet_pytorch import EfficientNet
from PIL import Image
from ipywidgets import IntProgress
from pytorchtools import EarlyStopping



augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(299),
                                               
                                                    torchvision.transforms.RandomCrop(299),                                                                            
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                                                    ])

def train_model( train_loader, val_loader , model , criterion , optimizer  , num_epochs , patience):
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    model.train()
    for epoch in range(num_epochs):
        
        print('Epoch : {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_accuracy = 0.0
        val_accuracy = 0.0
        train_loss = 0.0
        val_loss = 0.0
        
        model.train()
       
        for i ,data in enumerate(train_loader , 0):
            inputs , labels =data['image'] , data['label']
            inputs, labels = inputs.to(device , dtype=torch.float32), labels.to(device)
            optimizer.zero_grad()          
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(torch.sigmoid(outputs), 1)
            
            acc = torch.sum(pred==labels).item()
            train_accuracy += acc
            
            train_loss += loss.item()
            
        print("Train Loss: {}".format(train_loss / (i+1)))
        print("Train Accuracy: {}".format(train_accuracy / train_face.__len__()))
        print()
       
       
        model.eval()
        with torch.no_grad():
            for i ,data in enumerate(val_loader , 0):
                inputs = data['image']
                labels = data['label']
                inputs, labels = inputs.to(device , dtype=torch.float32), labels.to(device)
                ou = model(inputs)
                _, pred = torch.max(torch.sigmoid(ou), 1)
                
                val_accuracy += torch.sum(pred==labels).item()
                
                loss = criterion(ou, labels)
                val_loss += loss.item()
                
            print("val Loss: {}".format(val_loss / (i+1)))    
            print("val Accuracy: {}".format(val_accuracy / val_face.__len__()))
            print()
            
        early_stopping( (val_loss / (i+1)), model)
        if early_stopping.early_stop :
            print("Early stopping!!")
            break
# =============================================================================
#        if epoch==5:
#            torch.save(model.state_dict() , './EfficientNet/mask/3_5.pth')
#        if epoch==10:
#            torch.save(model.state_dict() , './EfficientNet/mask/3_10.pth')
# =============================================================================
           
    torch.save(model.state_dict() , './ensemble.pth')  
    return model

def makelist(path):
    l =[]
    for i in os.listdir(path):
        pp = os.path.join(path , i , '1.jpg')
        if os.path.exists(pp):
            l.append(pp)
    return l

def ffdata(fake_path , real_path):
    
    fake_type = ['Deepfakes','Face2Face','FaceShifter','FaceSwap','NeuralTextures']
    
    fake = []
    real = []
    for z in range(0,3):
        a = os.path.join(fake_path , fake_type[z] )
        for i in os.listdir(a):
            b = os.path.join(a , i )
            for j in range(1,11):
                c = os.path.join(b , str(j)+'.jpg' )
                fake.append(c)
    
    
    for i in os.listdir(real_path):
        b = os.path.join(real_path , i )
        for j in range(1,21):
            c = os.path.join(b , str(j)+'.jpg' )
            real.append(c)        
    
    face = maskdata(real , fake ,transforms = augmentation) 
     
    return face


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB._fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(2048+1536, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(torch.relu(x))
        
        return x


        
if __name__ =="__main__":

#----------------------xception----------------------------------------------    
    model = xception(pretrained=False)
    state_dict = torch.load(
        r'C:\Users\jeremy\Desktop\2021DF\model\FaceForensics-master\FaceForensics-master\classification\xception-b5690688.pth')
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    model.load_state_dict(state_dict)
    
    for param in model.parameters():
        param.requires_grad = False
    
    out_num = model.fc.in_features
    model.fc = nn.Linear(out_num,2)
    
# =============================================================================
#     model = Xception()
#     model.last_linear = model.fc  
#     out_num = model.fc.in_features
#     model.last_linear = nn.Linear(out_num,2)
#     del model.fc
#     
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     #summary(model , (3,256,256))
# =============================================================================
    
    model2 = EfficientNet.from_pretrained('efficientnet-b3' , num_classes=2)
    
    for param in model2.parameters():
        param.requires_grad_(False)   
    
    
    
#-------------------------efficient--------------------------------------------------    
# =============================================================================
#     #model = EfficientNet.from_pretrained('efficientnet-b3' , num_classes=2)
#     p = r'C:\Users\jeremy\Desktop\2021DF\model\FaceForensics-master\FaceForensics-master\classification\all_c23.p'
#     model = torch.load(p)
# =============================================================================
    #model = EfficientNet.from_pretrained('efficientnet-b3' , num_classes=2)
    
    
    model3 = MyEnsemble(model, model2)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model3 = model3.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer_ft = optim.Adam(model3.parameters(), lr=0.0001 )
    
    
    
    
      
# load dataset ===========================================================================
    
# =============================================================================
#     # celeb-df mask data
#     path = r'D:\Celeb-DF-v2\train\real_mask'
#     real1 = makelist(path)
#     path = r'D:\Celeb-DF-v2\train\yt-real_mask'
#     real2 = makelist(path)
#     real = real1 + real2
#     path = r'D:\Celeb-DF-v2\train\fake_mask'
#     fake = makelist(path)
#     
#     face = maskdata(real , fake ,transforms = augmentation)
#     
#     train_loader = DataLoader( face , batch_size = 8 , shuffle = True , num_workers = 0 )
# =============================================================================

    # FF++ data
    
    train_face = ffdata(fake_path = r'D:\FF++\data\train' , real_path = r'D:\FF++\data\train\original')
    val_face = ffdata(fake_path = r'D:\FF++\data\val' , real_path = r'D:\FF++\data\val\original')
    
    train_loader = DataLoader( train_face , batch_size = 8 , shuffle = True , num_workers = 0 )
    val_loader = DataLoader( val_face , batch_size = 8 , shuffle = True , num_workers = 0 )
    
    

    # celeb-df 以人為單位 
# =============================================================================
#     train_face = deepfakedata(0, 'D:/Celeb-DF-v2/train/real' , 'D:/Celeb-DF-v2/train/fake',transforms = augmentation ) 
#     
#     for i in range(1,50):
#         train_face += deepfakedata(i, 'D:/Celeb-DF-v2/train/real' , 'D:/Celeb-DF-v2/train/fake',transforms = augmentation )
#     
#     train_loader = DataLoader( train_face , batch_size = 8 , shuffle = True , num_workers = 0 )
#     
#     val_face = deepfakedata(10, 'D:/Celeb-DF-v2/train/real' , 'D:/Celeb-DF-v2/train/fake',transforms = augmentation ) 
#     
#     for i in range(51,60):
#         val_face += deepfakedata(i, 'D:/Celeb-DF-v2/train/real' , 'D:/Celeb-DF-v2/train/fake',transforms = augmentation )
#     
#     val_loader = DataLoader( val_face , batch_size = 8 , shuffle = True , num_workers = 0 )
# =============================================================================

    
    re = train_model(train_loader, val_loader , model3, criterion, optimizer_ft , patience = 25, num_epochs=60)    
    
    #summary(model , (3,256,256))
    
    
    
