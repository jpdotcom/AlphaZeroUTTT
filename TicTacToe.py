import math;
import numpy as np;
import random;
import time;

sz=3;
Xwins=[]
Owins=[]

for i in range(sz):

    Xwins.append((1<<(i*sz)) * ((1<<sz)-1)) #row wins 
    Xwins.append((1<<i) * ((1<<(sz*sz))-1) / ((1<<sz)-1)) #column wins 

Xwins.append(((1<<(sz*(sz+1)))-1)/((1<<(sz+1))-1)); #Main Diagnol win 
Xwins.append(((1<<(sz*(sz-1)))-1) * (1<<(sz-1)) / ((1<<(sz-1))-1)); #Other diagnol;

for i in range(len(Xwins)):
    Owins.append(Xwins[i] * ((1<<(sz*sz)) + 1));

for i in range(len(Xwins)):
    Xwins[i]=int(Xwins[i]);
    Owins[i]=int(Owins[i]);

def getBoard(value):
    out=[]
    for i in range(sz):
        row=[]
        for j in range(sz):
        
            val=getValueAt(value,i,j);
            if (val==1):
                row.append('O')
            if (val==-1):
                row.append('X');
            if (val==0):
                row.append("_")
        out.append(row);
    
    return out;

def display(value):

    board=getBoard(value);
    for row in board:
        print(row);
    return;

def place(value,r,c,mark):
    pos1=r*sz+c;
    if ((value>>pos1)%2==1 or gameOver(value)):
        return False; 
    else:
        value+=(1<<pos1);
        pos2=sz*sz+pos1;
        value+=(1<<pos2 if mark=="O" else 0);
        #LM=3*r+c;

        return value;

def getValueAt(value,r,c):

    #returns 1 if there is an O, 0 if there is nothing , and -1 if there is an X;

    pos1=r*sz+c;

    if ((value>>pos1)%2==0):
        return 0;
    else:
        pos2=sz*sz+pos1;
        
        return 1 if ((value>>pos2) % 2) else -1;

def checkWinner(value):
    #return 1 if O wins, 0 for nobody, and -1 for X
 
    for x in Xwins:
        if (((x&value)==x) and ((~(value>>(sz*sz))) & (x)==x)):
            
            return -1 
    for o in Owins:
        if ((o&value)==o):
            return 1
    return 0;


def getNextMoves(value,val):
    #return a list of serilazers of next possible states;
    if (checkWinner(value)!=0):
        return []
    out=[]
    for i in range(sz):
        for j in range(sz):
            if (getValueAt(value,i,j)==0):
                nextPos=value;
                nextPos=place(value,i,j,val);
                out.append((nextPos,3*i+j));    
    return out;

def gameOver(value):
    mask=(1<<(sz*sz))-1
    return (checkWinner(value)!=0  or value&mask==mask);

def getBoardMatrix(value):
    board=[[0 for i in range(sz)] for j in range(sz)]
    for i in range(sz):
        for j in range(sz):
            board[i][j]=getValueAt(value,i,j);
    return board
# value=0;
# player=0;
# while (True):
#     r,c=list(map(int,input().split()));

#     value=place(value,r,c,('O' if player else 'X'))
#     player^=1
#     display(value);
#     if (gameOver(value)):
#         print(checkWinner(value));
#         break;