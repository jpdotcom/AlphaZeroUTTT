import math;
import numpy as np;
import random;
import time;
from TicTacToe import *
#from model import *
from UTTT import UTTT;
from C4 import C4
from threading import Thread
from threading import Lock;
from collections import deque
import json
import torch.multiprocessing as mp
import torch
#from pit import battle;
from mcts import MCTS
import gc
from torchModel import Net,train
import os
import gzip
import torch_tensorrt
torch.backends.cudnn.benchmark=True
mp.set_start_method('spawn',force=True)
#multiprocessing.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


EPOCHS=4
BATCH_SIZE=2048;
DQSIZE=1320000;
TEMP_THRES=35; 
PSIZE=6; #How many processors
NUM_GAMES_PER_PROC=200;
TRAIN_START=200000;  #How many training examples before training starts
DIR_NOISE=False

pexmp=[0]*(PSIZE)
commonPTH="/home/jay/Desktop/AlphaUTTT(Torch)/Data"
Ppath="/home/jay/Desktop/AlphaUTTT(Torch)/ProcessExamples"
PTH="/home/jay/Desktop/AlphaUTTT(Torch)/model.pt"
def save_model(pth,model,opt):
    torch.save({
        
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        
        
        }, pth)
def load_model(pth):
    checkpoint = torch.load(pth);
    model=Net();
    model.load_state_dict(checkpoint['model_state_dict']);
    model.to(device=device);
    optimizer=torch.optim.Adam(model.parameters());
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
    return model,optimizer


def save(pth,examples,t='w',compress=False):
    data=examples
    

    if (compress):
        with gzip.open(pth, 'wt', encoding='UTF-8') as zipfile:
            json.dump(data, zipfile)         
    else:
        with open(pth,t) as file:
            json.dump(data,file);
def read(pth,decompress=False):
    
    if (decompress):
        with gzip.open(pth, 'rt', encoding='UTF-8') as zipfile:
            data = json.load(zipfile)
        return data        


    
    with open(pth,'r') as file:
        return json.load(file);
if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    #print(psutil. cpu_count())

    print(torch.cuda.is_available());
    print(torch.cuda.get_device_name(0))


    
        
    loadExamples=[]
    #print(len(read("examples.json",True)))
    #THREAD_COUNT = 5
    #results=[None]*THREAD_COUNT;
    
    
    mainModel=Net();
    optimizer=torch.optim.Adam(mainModel.parameters(),lr=0.00001,weight_decay=1e-4);
    mainModel.to(device=device)
    if (int(input("Load Model (0/1)? "))):

        mainModel,optimizer=load_model(PTH);
        for param_group in optimizer.param_groups:
            param_group["lr"]=0.00001;
    
        #print("HELLO?");
    print(optimizer)
    mainModel.share_memory();
    #loadExamples=read("examples.json",False);
    
    # train(mainModel,loadExamples,BATCH_SIZE,EPOCHS,optimizer);
    save_model(PTH,mainModel,optimizer);
   


#print(len(allExamples))
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#battle(mainModel,mainModel);
#@ray.remote(num_gpus=0.1)


def trainPlay(queue,model,RUNTIME=6):
    
    for trial in range(RUNTIME):
        tree=MCTS(model);
        #print("HELLO");
        SIM=200;
        examples=[]
        player=0;
        state=C4([0,0])
        #state=UTTT([0]*1,None)
        #state.display();
        numTurns=0;
        branching=0;
        while (not state.gameOver()):
            

            for i in range(SIM):
                #t=time.time();
                tree.select(tree.root,state);
                #print(time.time()-t);
                #t=time.time();

                #print(time.time()-t);
            
            example=[state.getNNinput('cpu')];
            
            #print(example.shape);
            # if (player):
            #     example[0]*=-1
            #     example[0][10]=abs(example[0][10]);
            if (player):
                example[0]*=-1

            #print(example[0]);
            temp=(numTurns<TEMP_THRES)
            if (temp):
                betterPolicy=np.array([tree.root.children[i].N for i in range(len(tree.root.children))])
            else:
                sims=[tree.root.children[i].N for i in range(len(tree.root.children))]
                msims=max(sims);
                sims=[(1 if (sims[i]==msims) else 0) for i in range(len(sims))]
            

                betterPolicy=np.array(sims);
                
                
            betterPolicy=betterPolicy/np.sum(betterPolicy);
        
            valid=state.getValidMoves();
            numTurns+=1;
            branching+=betterPolicy.shape[0]
            #print(betterPolicy.shape)
            nextNodes=tree.root.children;
            
            
            rootIndex=np.random.choice(betterPolicy.shape[0],p=betterPolicy)
            nextRoot=tree.root.children[rootIndex]
            actionNum=None;
            idx=0;
            
            for i in range(len(valid)):
                if (valid[i]):
                    valid[i]=betterPolicy[idx];
                    if (idx==rootIndex):
                        actionNum=i;
                    idx+=1 
                    
                else:
                    valid[i]=0;
            #print(idx==len(betterPolicy));

            betterPolicy=valid 
            
            #print(np.sum(betterPolicy))
            
            example.append(betterPolicy);
            example.append(tree.root.Q[0][0]);
            examples.append(example)
            tree.root=nextRoot
            state=state.nextState(actionNum,'X' if player else 'O')
            
            #state.display()
    
            
            tree.root.parent=None; # 4. gzip
            player^=1;
            #print(numTurns);
        reward=float(abs(state.checkWinner()));
        
        #print(reward);
        # for i in range(len(examples)-1,-1,-1):
        #     examples[i][2]=reward; 
        #     reward*=-1;

        print(branching/numTurns);
        #return examples;

        # print(idx);
        # print(len(pexmp))
        
        
        
    #print("HELLO");
    #print("DONE");
    return

@torch.no_grad()
def multi_train(num_games,model,fidx):
    rtModel=torch_tensorrt.compile(model,inputs=[torch_tensorrt.Input(min_shape=[1,2,9,9],opt_shape=[num_games,2,9,9],max_shape=[num_games,2,9,9])],enabled_precisions={torch.float})
    #rtModel=model
    PID=str(os.getpid());
    examplesToAdd=[]
    gamesLeft=num_games
    trees=[MCTS(DIR_NOISE) for i in range(num_games)];
    states=[UTTT([0 for j in range(10)],None) for i in range(num_games)];
    #states=[C4([0 for j in range(2)]) for i in range(num_games)]
    count=[0]*num_games
    gameEnded=[False]*num_games
    SIM=800;
    example_loc=[];
    winner=[0]*(num_games)
    # for i in range(num_games):

    # #model=ray.put(mainModel)
    #     t=time.time();
    #     examples=trainPlay();
    # #examples=ray.get([trainPlay.remote(model) for i in range(num_games)]);
    # #ray.shutdown();
    # #int(input(""));
    
    #     print("Training game #" + str(i+1) + " Done. Time taken: " + str(time.time()-t) + "s");
    #     examplesToAdd+=examples;
    
    # with open('examples.json','r') as file:

    #     loadExamples=json.load(file);
    # allExamples=deque(loadExamples+examplesToAdd,maxlen=DQSIZE);
    
    # data=[[np.array(example[0]).tolist(),np.array(example[1]).tolist(),np.array(example[2]).tolist()] for example in allExamples]
    # with open("examples.json",'w') as file:
    #     json.dump(data,file);
    # train(mainModel,data,BATCH_SIZE,EPOCHS);
    print("Training set started: "  + PID);
    t=time.time();
    simsDone=0;
    win,draw=0,0
    #gt=time.time() #time taken per round of simulations
    ##Parallelization... Batch multiple policy/value predicitons from multiple games via GPU
    while (gamesLeft):  
        batch=[]
        treeIdx=[]
        nodes=[]
        stateExpand=[]
        for i in range(num_games):
            tree=trees[i];
            state=states[i];
            over=gameEnded[i]
            root=tree.root;
            
            if (not over):
                node,expandState=tree.select(root,state);
                if (node!=None):
                    nnInput=expandState.getNNinput();
                    nnInput[0]=(nnInput[0]*-1 if node.player else nnInput[0])
                    
                    batch.append(nnInput)
                    treeIdx.append(i);
                    nodes.append(node);
                    stateExpand.append(expandState);
        if (len(batch)!=0):
            #print(len(batch));
            with torch.no_grad():
                
                batch=torch.stack(batch);
                batch.to(device);
                #print(batch.dtype)
                policies,values=rtModel(batch);
                #print("WAIT: ")
               # print(input())
                policies=policies.to("cpu").detach().numpy();
                values=values.to("cpu").detach().numpy();  
            #print(policies.shape[0]==len(treeIdx))      
            for (i,idx) in enumerate(treeIdx):
                tree=trees[idx];
                state=stateExpand[i];
                node=nodes[i];
                #print(policies[i].shape)
                tree.expand(node,state,policies[i],values[i][0]);
                tree.propogate(node,values[i][0]);
                if (DIR_NOISE and node==tree.root):
                    tree.get_noise_param();
        simsDone+=1;

        if (simsDone==SIM):

            for j in range(num_games):
            
                if (not gameEnded[j]):
                    tree=trees[j];
                    state=states[j];
                    over=gameEnded[j]
                    root=tree.root;
                    numTurns=count[j];

                    example=[state.getNNinput('cpu')];
                    player=root.player
                    if (player):
                        example[0]*=-1
                
                    #print(example[0]);
                    temp=(numTurns<TEMP_THRES)
                    if (temp):
                        betterPolicy=np.array([tree.root.children[i].N for i in range(len(tree.root.children))])
                    else:
                        sims=[tree.root.children[i].N for i in range(len(tree.root.children))]
                        msims=max(sims);
                        sims=[(1 if (sims[i]==msims) else 0) for i in range(len(sims))]
                    

                        betterPolicy=np.array(sims);
                        
                        
                    betterPolicy=betterPolicy/np.sum(betterPolicy);
                    #print(betterPolicy.shape)
                    valid=state.getValidMoves();
                    
                    #branching+=betterPolicy.shape[0]
                    #print(betterPolicy.shape)
                    nextNodes=tree.root.children;
                    #print(betterPolicy.shape[0]);
                    #print(betterPolicy)
                    rootIndex=np.random.choice(betterPolicy.shape[0],p=betterPolicy)
                    nextRoot=tree.root.children[rootIndex]
                    actionNum=None;
                    idx=0;
                    #print(rootIndex);
                    #print(valid);
                    # print(valid);
                    # print(betterPolicy)
                    # if (sum(valid)!=len(betterPolicy)):
                    #     state.display()
                    #     print(state.prevPos)
                    #     print(valid);
                    #     print(betterPolicy)
                    # print(len(tree.root.children))
                    # print(len(tree.root.children)==betterPolicy.shape[0])
                    # print(rootIndex)
                    for i in range(len(valid)):
                        if (valid[i]):
                            valid[i]=betterPolicy[idx];
                            #print(idx);
                            if (idx==rootIndex):
                                actionNum=i;
                            idx+=1 
                            
                        else:
                            valid[i]=0;
                    
                    #print(idx==len(betterPolicy));

                    betterPolicy=valid 
                    
                    #print(np.sum(betterPolicy))
                    example.append(betterPolicy);
                    example.append(-1 if player else 1);
                    #print(example[0].device)

                    examplesToAdd.append(example)
                    tree.root=nextRoot
                    if (DIR_NOISE):
                        tree.get_noise_param();
                    state=state.nextState(actionNum,'X' if player else 'O')
                    example_loc.append(j);
                    #state.display()

                    states[j]=state;
                    tree.root.parent=None;
                    #(tree.root.state.display());
                    #print("=========================")
                    if (state.gameOver()):
                        #state.display();=
                        gameEnded[j]=True;
                        gamesLeft-=1


                        winner[j]=float(state.checkWinner());
                        if (abs(winner[j])):
                            win+=1
                        else:
                            draw+=1;
                        print("Training game #" + str(j+1) + " Done : " + PID);
                        #state.display()
                    count[j]+=1;
            simsDone=0;
           
            #print(time.time()-gt);
            #gt=time.time();

            #print(gamesLeft)
            #print(states[0].games==states[1].games)
    print("Training games done for pid: " + PID + ". Time taken: " + str(time.time()-t) + "s. Total wins: " + str(win) + ". Total draws: " + str(draw));
    for i in range(len(examplesToAdd)):
        examplesToAdd[i][2]*=winner[example_loc[i]]
    examplesToAdd=[[np.array(example[0]).tolist(),np.array(example[1]).tolist(),np.array(example[2]).tolist()] for example in examplesToAdd]    
    pth=Ppath+"/P"+str(fidx)+".json"
    save(pth,examplesToAdd);
    print("Data saved to: " + pth)
    return examplesToAdd;
if __name__ == '__main__':

    
    print("Program began. The data will be saved here: \n"+ "\n".join([commonPTH+"/P"+str(i+1)+".json" for i in range(PSIZE)]));
    v=False; #set to False if there are zero examples on the json file
    orig=NUM_GAMES_PER_PROC
    NUM_GAMES_PER_PROC=600;
    files=[int(name[:-5]) for name in os.listdir(commonPTH)]  
    offset=0

    if (len(files)>0):

        min_val=min(files);
        offset=min_val
    sz=len(files);
    print(f"OFFSET (FOR DEQUE): {offset}")
    
    files=0;
    print("DATASET SIZE: " + str(sz))
    #train(mainModel,sz,BATCH_SIZE,10,offset,torch.optim.Adam(mainModel.parameters()));
    #save_model(PTH,mainModel,optimizer);
    #train(mainModel,sz,BATCH_SIZE,EPOCHS,offset,optimizer);
    #save_model(PTH,mainModel,optimizer)
    for i in range(5000):
        mainModel.eval();
       
        #print(model_int8)
        #rtModel=torch_tensorrt.compile(mainModel,inputs=[torch_tensorrt.Input(min_shape=[1,2,9,9],opt_shape=[NUM_GAMES_PER_PROC,2,9,9],max_shape=[NUM_GAMES_PER_PROC,2,9,9])],enabled_precisions={torch.half})

        print("Starting Processes for " + str(i) + "th time...")
        processes=[]
        t=time.time();
        
        for i in range(PSIZE):
            
            p=mp.Process(target=multi_train,args=(NUM_GAMES_PER_PROC,mainModel,i+1))
            p.start();
            processes.append(p)
        
        for p in processes:
            p.join();
        for p in processes:
            p.terminate();
   
        print("Processes Terminated");
        
        #loadExamples=read("examples.json",v);
        v=False;
        for i in range(PSIZE):
            examples=read(Ppath+"/P"+str(i+1)+".json")
            for j in range(len(examples)):
                while (sz>=DQSIZE):
                    os.remove(commonPTH+"/"+str(offset)+".json")
                    offset+=1
                    sz-=1
                save(commonPTH+"/"+str(sz+offset)+".json",examples[j])
                sz+=1
        if (sz>=TRAIN_START):
            train(mainModel,sz,BATCH_SIZE,EPOCHS,offset,optimizer);
        if (sz==DQSIZE):
            NUM_GAMES_PER_PROC=orig
        save_model(PTH,mainModel,optimizer);
        
        # if (i==0):
        #     torch.save(mainModel.state_dict(),PTH)
        #     continue;
        # else:
        #     oldModel=Net();
        #     oldModel.load_state_dict(torch.load(PTH))
        #     oldModel.to(device=device);
        #     better=battle(oldModel,mainModel)
        #     if (better):
        #         torch.save(mainModel.state_dict(),PTH)
        #     continue;
        
        data=[]
        allExamples=[]
        loadExamples=[];
        _=gc.collect();
# tree=MCTS();
# #print(tree.root.state.checkWinner());200
#     print('------------------------------')
