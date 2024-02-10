import numpy as np;
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OFFSET=(1<<6) + (1<<13) + (1<<20) + (1<<27) + (1<<34) + (1<<41) + (1<<48);
FCOL=(1<<6)-1;
class C4:

    def __init__(self,values) -> None:
        self.board=values 
    
        
    def place(self,row,val):

        idx=1 if val=='X' else 0;
        bit_pos=7*row;

        for i in range(6):
            if (not self.occupied(bit_pos+i)):
                self.board[idx]+=(1<<(bit_pos+i));
                return;         
    
    def occupied(self,pos):

        return (self.board[0]>>pos)&1 or (self.board[1]>>pos)&1;
    def gameOver(self):
        return (self.checkWinner()!=0 or (self.board[0] + self.board[1] + OFFSET)==(1<<49)-1);
    def check(self,m):

        y=m & (m>>(7));
        if (y & (y>>(2*7))):
            return True 
        
        y=m&(m>>6);
    
        if (y & (y>>(2*6))):
            return True
        
        y=m & (m>>8);
    
        if (y & (y>>(2*8))):
            return True 
        
        y= m & (m>>1);
        if (y & (y>>2)):
            return True 
        
        return False
    def getCharAt(self,pos):

        return "O" if (self.board[0]>>pos)&1 else ("X" if (self.board[1]>>pos)&1 else "_")
    def checkWinner(self):
        return (self.check(self.board[0]) - self.check(self.board[1]))

    def nextState(self,action,mark):

        copyC4=C4([self.board[0],self.board[1]]);
        copyC4.place(action,mark);
        return copyC4

    # int64_t y = board & (board >> 7);
    # if (y & (y >> 2 * 7)) // check \ diagonal
    #     return true;
    # y = board & (board >> 8);
    # if (y & (y >> 2 * 8)) // check horizontal -
    #     return true;
    # y = board & (board >> 9);
    # if (y & (y >> 2 * 9)) // check / diagonal
    #     return true;
    # y = board & (board >> 1);
    # if (y & (y >> 2))     // check vertical |
    #     return true;
    # return false;

    def display(self,supressOut=False):
        out=[['_' for j in range(7)] for i in range(6)]
        for i in range(6):

            for j in range(7):

                out[5-i][j]=self.getCharAt(i+7*j);
        if (supressOut):
            return out;
        
        for row in out:
            print(row)
        return out;
    def getNNinput(self,device=device):

        stringBoard=self.display(True);

       

        out=[[0 for j in range(7)] for i in range(6)]

        for i in range(6):
            for j in range(7):
                out[i][j]=(1 if stringBoard[i][j]=='O' else (-1 if stringBoard[i][j]=='X' else 0))
        
        return torch.tensor([out],dtype=torch.float32,device=device);

    def getValidMoves(self):

        s=self.board[0]+self.board[1];
        out=[False]*7;
        for i in range(7):
            out[i]=not ((s&FCOL)==FCOL)
            s>>=7;
        return out;

            

# game=C4([0,0]);
# val=0;
# while (not game.gameOver()):

#     game.place(int(input()),'X' if val else 'O');
#     print(game.getNNinput())
#     print(game.getValidMoves())
#     val^=1;

    
# print(game.checkWinner())
