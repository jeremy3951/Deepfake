
from data import maskdata

from Xcept import xception , Xception
import time
import torch
from torchvision import datasets, models, transforms
import os
import torch.nn as nn

from sklearn.ensemble import AdaBoostClassifier

import copy
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader ,Dataset , random_split

import numpy as np
import joblib
from efficientnet_pytorch import EfficientNet
from PIL import Image

augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(299),
                                                    torchvision.transforms.RandomCrop(299),                                                                            
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                                                    ])


class testdata(Dataset):
    def __init__(self ,Dir ,label ,transforms):
        self.dir = Dir
        self.label = label
        self.transforms = transforms
           
    def __len__(self):   #抓幾張圖片

        return 50   # idx 從 0 開始算
    
    def __getitem__(self  , idx ):   # 弄好 (img , label)
        
        if os.path.exists(self.dir+'/'+str(idx+1)+'.jpg'):
            pic = Image.open(self.dir+'/'+str(idx+1)+'.jpg').convert('RGB')
            if self.transforms is not None:
                pic = self.transforms(pic)
        
            sample = {'image': pic , 'label': self.label}  
            return sample
        else:
            print('file not found')
            print(self.dir+'/'+str(idx+1)+'.jpg')
            return 0


def test(loader, model):
    model.eval()
    accuracy = 0.
    with torch.no_grad():
        for i,data in enumerate(loader , 0):
            inputs = data['image']
            labels = data['label']
            inputs, labels = inputs.to(device , dtype=torch.float32), labels.to(device)
            ou = model(inputs)
            _, pred = torch.max(torch.sigmoid(ou), 1)
            
            print()
            print(pred)
            print(labels)
            print()
            accuracy += torch.sum(pred==labels).item()
        
    
    print("Test Accuracy: {}".format(accuracy / len(face)))
    
    f = open('./res.txt' , 'w')
    f.write("Test Accuracy: {}".format(accuracy / len(face)))
    f.close()
    


def celeb_sum (path):   # 加總一個目錄下所有的  
    l = []
    for i in os.listdir(path):
        l.append(i)
    
    for i in range(0,518):
        
        if i==0:
            if(len(l[i])>10):
                face = testdata(path+'/'+l[i] , 1 , transforms = augmentation)
            else:
                face = testdata(path+'/'+l[i] , 0 , transforms = augmentation)
        
        else:
            if(len(l[i])>10):
                face += testdata(path+'/'+l[i] , 1 , transforms = augmentation)
            else:
                face += testdata(path+'/'+l[i] , 0 , transforms = augmentation)
        
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

def ff(fake_path , real_path):
    #fake_type = ['FaceShifter','NeuralTextures']
    
    fake_type = ['Deepfakes','Face2Face','FaceSwap']
    
    fake = []
    real = []
    for z in range(0,3):
        a = os.path.join(fake_path , fake_type[z] )
        for i in os.listdir(a):
            b = os.path.join(a , i )
            for j in range(1,41):
                c = os.path.join(b , str(j)+'.jpg' )
                fake.append(c)
    
    
    for i in os.listdir(real_path):
        b = os.path.join(real_path , i )
        for j in range(1,41):
            c = os.path.join(b , str(j)+'.jpg' )
            real.append(c)        
    
    face = maskdata(real , fake ,transforms = augmentation) 
     
    return face

if __name__ =="__main__":
    
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
    
    model2 = EfficientNet.from_pretrained('efficientnet-b3' , num_classes=2)
    
    for param in model2.parameters():
        param.requires_grad_(False)  
        
    model3 = MyEnsemble(model, model2)
    
    
    
# =============================================================================
#     model = xception(pretrained=False)
#     model.fc = model.last_linear
#     del model.last_linear
#     state_dict = torch.load(
#         r'C:\Users\jeremy\Desktop\2021DF\model\FaceForensics-master\FaceForensics-master\classification\xception-b5690688.pth')
#     for name, weights in state_dict.items():
#         if 'pointwise' in name:
#             state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
#     model.load_state_dict(state_dict)
#     model.last_linear = model.fc
#     out_num = model.fc.in_features
#     model.last_linear = nn.Linear(out_num,2)
#     model.load_state_dict(torch.load('./XceptionNet/mask2/m1.pth'))
# =============================================================================

# =============================================================================
#     model = Xception()
#     model.last_linear = model.fc  
#     out_num = model.fc.in_features
#     model.last_linear = nn.Linear(out_num,2)
#     del model.fc
# =============================================================================
    
    #model.load_state_dict(torch.load('XceptionNet/m2.pth'))
    model3.load_state_dict(torch.load('ensemble.pth'))

# =============================================================================
#     p = r'C:\Users\jeremy\Desktop\2021DF\model\FaceForensics-master\FaceForensics-master\classification\all_c23.p'
#     q = r'C:\Users\jeremy\Desktop\2021DF\model\XceptionNet\mask\m2.pth'
#     model = torch.load(p)
#     model.load_state_dict(torch.load(q))
# =============================================================================
      
    #model.load_state_dict(torch.load('./EfficientNet/mask/3_5.pth'))
    
                          
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model3 = model3.to(device)
    
    
# loaddata ===========================================================================
    
#----------------------------celceb-df------------------------------------------
    path = 'D:/Celeb-DF-v2/test'
    face = celeb_sum(path)
    
#-----------------------------FF++-----------------------------------------
         
    #face = ff(fake_path = r'D:\FF++\data\test',real_path = r'D:\FF++\data\test\original') 
    #summary(model , (3,299,299))
    

    
    test_loader = DataLoader( face , batch_size = 16 , shuffle = True , num_workers = 0 )
    
    #test(test_loader , model3)   







