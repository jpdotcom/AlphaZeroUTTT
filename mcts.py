import math;
import numpy as np;
import random;
import time;
from TicTacToe import *
#from model import *
from UTTT import UTTT;
from threading import Thread
from threading import Lock;
from collections import deque
import json
import multiprocessing
from multiprocessing import Process
#import tensorflow as tf
import traceback
import torch;
import torch.functional as F
import ray
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
torch.backends.cudnn.benchmark=True


ALPHA=1
eps=0

class Node:
    
    def __init__(self,p,player,action=None) -> None:
        self.player=player;
        self.N=0;
        self.Q=0;
        self.P=0;
        self.parent=p;
        self.children=[];
        self.c=4;
        #self.v=0;
        
        #state of the game
        #self.state=s;
        self.actionDone=action
    def value(self,noise=None):
        P=self.P 
        if (noise!=None):
            P=(1-eps)*P + eps*noise;
        Q=self.Q
        if (self.parent==None):
            print("HELLO?");
        U=self.c * P * math.sqrt(self.parent.N)/(1+self.N)
        #print(f"Q_value: {Q}. U_value: {U}. Noise added: {noise!=None}")
        
        return -1*(Q) + U

    def add(self,child) -> None:
        
        self.children.append(child);


class MCTS:

    def __init__(self,train=True) -> None:
        self.root=Node(None,0);
        self.move=None;
      
        self.noise=None
        self.train=train
        #self.repeated={tuple([0]*10) : self.root}
    @torch.no_grad()
    def get_noise_param(self):
        n=len(self.root.children);
   
        dir_vector=np.random.dirichlet([ALPHA]*n);
   
        self.noise=dir_vector
     
        return
    @torch.no_grad()
    def select(self,node,state):

     
        
        while (len(node.children)>0):
            
            try:
                #UCB=[node.children[i].value(self.noise[i] if self.root==node and self.train else None) for i in range(len(node.children))]

                maxUCB=-float('inf');
                nextNode=None;
                allMaxUCB=[]
                for i in range(len(node.children)):
                    UCB_val=node.children[i].value(self.noise[i] if self.root==node and self.train else None)
                    if (UCB_val>maxUCB):
                        maxUCB=UCB_val;
                        nextNode=node.children[i];

                #nextNode=random.choice(allMaxUCB);
                
               
                nextState=state.nextState(nextNode.actionDone,'X' if node.player else 'O');

                
                
                node=nextNode
                state=nextState
            
            except Exception:
                #for child in node.children:
                    #print(child);
                #input("WHAT");
                traceback.print_exc()
                state.display();
                print([child.Q for child in node.children])
                #print("ERROR");
        
        #print(node.value());
        if (state.gameOver()):
            value=state.checkWinner();
            value=-1*abs(value);
            self.propogate(node,value);
            return None,None

        else:
            #value=self.expand(node,state);
            return node,state
        
        #state=None;

       
        return;
    @torch.no_grad()
    def propogate(self,node,value):

        
        while (node!=None):
            node.Q=(node.Q*node.N+value)/(node.N+1);
            node.N+=1
            value*=-1;
            node=node.parent; 
        return;
    @torch.no_grad()
    def expand(self,node,state,P,value):


        #nextStates=state.getNextMoves('X' if node.player else 'O');
        # nninput=[state.getNNinput()];
        # nninput=torch.stack(nninput);
        # # if (node.player==1):
        # #     nninput*=-1;softmax
        # #     nninput[10]=abs(nninput[10]);
        # if (node.player==1):
        #     nninput[0]  *=-1;
         
        #print(nninput);
        #state_tuple=tuple(node.state.games) + tuple([node.state.prevPos]);
        #if (state_tuple in self.cachedNN):
            #P,value=self.cachedNN[state_tuple]
           
            #print(len(node.children))
        #else:
        # P,value=self.model(nninput);

        # P=P.to('cpu').detach().numpy();
  
        # value=value.to('cpu').detach().numpy();
        
        P_exp=np.exp(P);
      
        P=P_exp/(np.sum(P_exp));
     
        #print(np.sum(P));
        #self.cachedNN[state_tuple]=[P,value];

        valid=np.array(state.getValidMoves());

        
        P*=valid;
        P_sum=np.sum(P);
        
        #print(P_sum);
        if (P_sum<=0):
            print("P SUM ZERO");
            print("Assigning Equal Probabilities...")
            P=valid;
            P_sum=np.sum(valid);
        P=P/(P_sum)
        
        #print(np.max(P));
        #if (node==self.root):
            #print(nextStates[2].games[0].lastMove())
        
        
    
        
        for idx,val in np.ndenumerate(valid):
        
            if (val):
        #print(state.games[0].lastMove())
        # tState=tuple(state.games);
        # if (tState in self.repeated):
        #     node.children.append(self.repeated[tState])
        #     #print("HELLO");
        # else:
        #     childNode=Node(node,state,node.player^1);
        #     self.repeated[tState]=childNode
        #     node.children.append(childNode);
        
                child=Node(node,node.player^1,idx[0])
                #print(idx);
                child.P=P[idx[0]]

                
                node.children.append(child)
            
        # if (len(node.children)!=np.sum(valid)):
        #     print(input("WRONG"));
        return value;
    
    # def simulate(self,node):
    #     state=node.state;
    #     nextPlayer='X' if node.player else 'O'
        
    #     while (not state.gameOver()):
            
           
    #         state=random.choice(state.getNextMoves(nextPlayer));
    #         if (nextPlayer=='X'):
    #             nextPlayer='O';
    #         else:
    #             nextPlayer='X';
        
    #     return state;

    # def getUserInput(self,lock):
    #     user_input = input('Move: ')
    #     with lock:ƒ
    #         self.move = list(map(int,user_input.split()));
        
    
    # def userInputBasedSimulation(self,lock):
    #     #run simulations while waiting for user input 

    #     while (True):
    #         with lock:
    #             if(self.move is not None):
    #                 print("Terminating userInputBasedSimulation...")
    #                 return
    #         node=self.select();
    #         state=self.simulate(node);
    #         self.propogate(state,node);

    # def propogate(self,state,node,reward):

    #     winner=state.checkWinner();
        
    #     while (node!=None):
            
    #         node.Q=((node.Q)*(node.N) + reward)/(node.N+1)
    #         node.N+=1;
            
            
    #         reward*=-1;
    #         node=node.parent;
    #         #print("HELLO");
    
    # def play(self,player,LIMIT=10):
    #     if (self.root.state.gameOver()):
    #         return False

    #     if (player==0):
    #         timeStart=time.time()
    #         while (time.time()-timeStart<LIMIT):
                
    #             node=self.select();
    #             state=self.simulate(node);
    #             self.propogate(state,node);
    #         nextNodes=[self.root.children[i] for i in range(len(self.root.children))]

    #         maxSim=max([node.N for node in nextNodes])
    #         allMaxSim=[]

    #         for node in nextNodes:
    #             if (node.N==maxSim):
    #                 allMaxSim.append(node)
    #         #print(self.root.N)
    #         self.root=random.choice(allMaxSim)
    #         self.root.parent=None;
            

    #     else:
    #         lock = Lock();
            
    #         user_input_thread = Thread(target=self.getUserInput, args=(lock,))
    #         simulation_thread = Thread(target=self.userInputBasedSimulation,args=(lock,))
            
    #         user_input_thread.start()
    #         simulation_thread.start()

    #         user_input_thread.join()
    #         simulation_thread.join()
            
    #         # if (len(self.root.children)==0):
    #         #     self.exapnd(self.root);

    #         #print(len(self.repeated))
    #         node=None;
    #         gamePos=3*self.move[0]+self.move[1]
    #         nextValue=place(self.root.state.games[gamePos],self.move[2],self.move[3],"X")
    #         for child in self.root.children:
    #             #print(child.state.games[gamePos]);
    #             if (child.state.games[gamePos]==nextValue):
    #                 node=child;
    #         self.root=node;
    #         #self.root=self.move;
    #         self.move=None;
    #         #self.root=random.choice(self.root.children);
    #     self.root.state.display();
    #     return True