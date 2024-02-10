from TicTacToe import *
from copy import deepcopy
import numpy as np;
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UTTT:

    def __init__(self,vals,ppos=None) -> None:
        self.games=vals;
        #print(vals[1].lastMove())
        self.prevPos=ppos
        
    def place(self,pos,mark):
        
        
        gameNum=3*pos[0]+pos[1];
        
        valid=place(self.games[gameNum],pos[2],pos[3],mark);
        
        if (not valid):
            return False;
        
        self.games[gameNum]=valid;
        if (gameOver(self.games[gameNum])):
            v=checkWinner(self.games[gameNum])
            
            if (v!=0): #make sure it is not a draw. Don't need to check who it is because last player always wins if they didn't draw.
                self.games[9]=place(self.games[9],pos[0],pos[1],mark);
        
        newPrevPos=3*pos[2]+pos[3];
    
        if (gameOver(self.games[newPrevPos])):
            self.prevPos=None
        else:
            self.prevPos=newPrevPos


        # valid=place(self.games[0],pos[0],pos[1],mark);
        # if (not valid):
        #     return False;
        # self.games[0]=valid;
    def nextState(self,action,mark):
        #print(action);
        cgames=[g for g in self.games]
        copyUTTT=UTTT(cgames,self.prevPos);
        pos=[]
        for i in range(4):
            pos.append(action%3);
            action//=3; 
        pos.reverse();
        copyUTTT.place(pos,mark);


        # cgames=[g for g in self.games]
        # copyUTTT=UTTT(cgames,self.prevPos);
        # pos=[]
        # for i in range(2):
        #     pos.append(action%3);
        #     action//=3; 
        # pos.reverse();
        # #print(pos);
        # copyUTTT.place(pos,mark);
        
        return copyUTTT
    def gameOver(self):
        allMinisDone=True;
        for i in range(9):
            allMinisDone=allMinisDone and gameOver(self.games[i])
        return (gameOver(self.games[9]) or allMinisDone)
        # allMinisDone=True;
        # for i in range(1):
        #     allMinisDone=allMinisDone and gameOver(self.games[i])
        # return (gameOver(self.games[0]) or allMinisDone)
    def checkWinner(self):
        return checkWinner(self.games[9]);
        #return checkWinner(self.games[0]);

    def getNextMoves(self,mark):
        nextStates=[]

        if (self.gameOver()):
            return []
        
        if (self.prevPos==None):

            for i in range(9):
                g=self.games[i];
                nextGMoves=getNextMoves(g,mark);
                for (nextPos,movePlayed) in nextGMoves:
                    nextState=[self.games[j] for j in range(0,i)]
                    nextState.append(nextPos)
                    nextState=nextState + [self.games[j] for j in range(i+1,10)]
                    if (checkWinner(nextPos)!=0):
                        nextState[9]=place(nextState[9],i//3,i%3,mark);
                    newLastMove=None if (gameOver(nextState[movePlayed])) else movePlayed;
                    nextStates.append(UTTT(nextState,newLastMove));
        else:
            i=self.prevPos;
            g=self.games[i];
            nextGMoves=getNextMoves(g,mark);
            for (nextPos,movePlayed) in nextGMoves:
                nextState=[self.games[j] for j in range(0,i)]
                nextState.append(nextPos)
                nextState=nextState + [self.games[j] for j in range(i+1,10)]
                if (checkWinner(nextPos)!=0):
                    nextState[9]=place(nextState[9],i//3,i%3,mark);
                newLastMove=None if (gameOver(nextState[movePlayed])) else movePlayed;
                nextStates.append(UTTT(nextState,newLastMove));
        return nextStates;
    def display(self):

        for game in self.games:
            display(game);
            print("=======================")

    def getNNinput(self,device=device):
        
        out=[]
        img1=[]
        for i in range(9):

            img1.append(getBoardMatrix(self.games[i]));
        img1=np.array(img1);
        img1=img1.reshape((3,3,3,3));
        img1=np.transpose(img1,(0,2,1,3));
        img1=img1.reshape((9,9))
        
        
        img2=np.array(self.getValidMoves()).reshape(9,3,3);
        img2=img2.reshape((3,3,3,3));
        img2=np.transpose(img2,(0,2,1,3));
        img2=img2.reshape((9,9))
        out=torch.tensor(np.array([img1,img2]),dtype=torch.float,device=device);
        return out;
        


        # for i in range(1):

        #     out.append(getBoardMatrix(self.games[i]));
        # # if (self.prevPos==None):
        # #     out.append(getBoardMatrix(0))
        # # else:
        # #     fakeGame=place(0,self.prevPos//3,self.prevPos%3,"O");
        # #     out.append(getBoardMatrix(fakeGame));
        # 
        # #print(out.shape);
        #return out;
            

    def getValidMoves(self):
        out=[False]*81;
        idx=0;
        if (self.prevPos==None):

            for i in range(9):
                
                for j in range(3):
                    for k in range(3):

                        val=self.games[i];
                        out[idx]=place(val,j,k,"O")!=False 
                        idx+=1
        else:
            idx=9*self.prevPos
                
            for j in range(3):
                for k in range(3):

                    val=self.games[self.prevPos];
                    out[idx]=place(val,j,k,"O")!=False 
                    idx+=1
        # out=[False]*9;
        # idx=0;
        # for i in range(1):
            
        #     for j in range(3):
        #         for k in range(3):

        #             val=self.games[i];
        #             out[idx]=place(val,j,k,"O")!=False 
        #             idx+=1
        return out;