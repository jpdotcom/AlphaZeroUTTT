import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import random
import numpy as np;
import gzip
import json
import torch_tensorrt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_hidden=256
hidden_layers=15
game_row=3
game_col=3
game_out=81
filter_in=2
init_stride=3;
DATAPTH="/home/jay/Desktop/AlphaUTTT(Torch)/Data"
torch.backends.cudnn.benchmark=True
def read(pth,decompress=False):
    
    if (decompress):
        with gzip.open(pth, 'rt', encoding='UTF-8') as zipfile:
            data = json.load(zipfile)
        return data        


    try:

        with open(pth,'r') as file:
            return json.load(file);
    except:
        return False
class ResBlock(nn.Module):
    def __init__(self):

        super().__init__();
        self.conv1=nn.Conv2d(num_hidden,num_hidden,kernel_size=(3,3),padding=1,bias=False);
        self.conv2=nn.Conv2d(num_hidden,num_hidden,kernel_size=(3,3),padding=1,bias=False);
        self.BN1=nn.BatchNorm2d(num_hidden);
        self.BN2=nn.BatchNorm2d(num_hidden);
    def forward(self,x):

        res=x; 
        out=F.relu(self.BN1(self.conv1(x)));
        out=self.BN2((self.conv2(x)));
        out+=res; 
        out=F.relu(out);
        return out;

class Net(nn.Module):
    def __init__(self):
        super().__init__();

        # self.conv1=nn.Conv2d(1,700,kernel_size=(2,2));
        # self.bn1=nn.BatchNorm2d(700)
        # self.conv2=nn.Conv2d(700,700,kernel_size=(2,2));
        # #self.bn2=nn.BatchNorm2d(num_hidden);
        # 
       
        self.start=nn.Conv2d(filter_in,num_hidden,kernel_size=(3,3),stride=init_stride,padding=1);
        self.BN=nn.BatchNorm2d(num_hidden);
        self.resNet=nn.ModuleList([ResBlock() for i in range(hidden_layers)])
        #self.flatten=nn.Flatten();
        #self.linear1=nn.Linear(num_hidden*3*3,1000);
  
        #self.flatten=nn.Flatten();
        self.policy=nn.Sequential(
            nn.Conv2d(num_hidden,num_hidden,kernel_size=(1,1)),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_hidden*game_row*game_col,game_out),
            
        )

        self.value=nn.Sequential(
            nn.Conv2d(num_hidden,3,kernel_size=(1,1)),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*game_row*game_col,num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden,1),
            nn.Tanh()
        )
        #self.value2=nn.Linear(150,1)
    def forward(self,x):
        
        x=F.relu(self.BN(self.start(x)));
        for resBlock in self.resNet:
            x=resBlock(x);
        #print(x.shape);
        #x=self.flatten(x)
        policy=self.policy(x);
        value=self.value(x);

        return policy,value;

def train(model,sz,batchSize,epochs,offset,optimizer=None):
    model.train();
    policyLoss=nn.CrossEntropyLoss();
    valueLoss=nn.MSELoss();
        
    for param_group in optimizer.param_groups:
        print(param_group["lr"])
    #random.shuffle(examples);
    for epoch in range(epochs):
        totalPLoss=0;
        totalVLoss=0
        batchesTrained=0;
        
        for batchIdx in range(0,sz,batchSize):
            ids=np.random.randint(offset,sz+offset,size=batchSize)
            batch=[]
            for id in ids:
                example=read(DATAPTH+"/"+str(id)+".json")
                if (example!=False):
                    batch.append(example);
            
            #batch=[examples[id] for id in ids]
            #batch=list(itertools.islice(examples,batchIdx,min(batchIdx+batchSize,len(examples))))
            optimizer.zero_grad()
            train=[example[0] for example in batch];
            train=torch.tensor(train,device=device)
            
            ptrain=[example[1] for example in batch];
            ptrain=torch.tensor(ptrain,device=device)
            
            vtrain=[example[2] for example in batch]
            vtrain=torch.tensor(vtrain,device=device);
            vtrain=torch.reshape(vtrain,[len(batch),1]);
            
            pout,vout=model(train);
            l1=policyLoss(pout,ptrain);
            l2=valueLoss(vout,vtrain)
            loss= l1+l2;
            totalPLoss+=l1.item();
            totalVLoss+=l2.item();
            loss.backward();
            
            optimizer.step();
            batchesTrained+=1
        print("Epoch " + str(epoch+1) +  " Completed. Average Policy Loss : " + str(totalPLoss/batchesTrained) + ". Average Value Loss: " + str(totalVLoss/batchesTrained));
    
    return;


