from mcts import MCTS 
import random
from UTTT import UTTT
import torch
from torchModel import Net
from C4 import C4
import torch_tensorrt
NUM_GAMES=1
SIM=2500;
THRESHOLD=0.5
def battle(model1,model2):

    t1=MCTS(False);
    t2=MCTS(False);


    win,loss,draw=[0,0,0]
    for i in range(NUM_GAMES):
        
        players=[t1,t2];
        swapped=False; 
        state=UTTT([0]*10,None);
        #state=UTTT([0]*1,None);
        v=random.uniform(0,1)
        print(v);
        if (v<0.5):
            players=[t2,t1];
            swapped=True;
            print("New Model is playing O");
        else:
            print("New Model is playing X")
        idx=0;
        while (not state.gameOver()):
            
            for j in range(SIM):
                #node,value=t1.select();
                #t1.propogate(node.state,node,value);

                #node,value=t2.select();
                #t2.propogate(node.state,node,value);
               
                node,expand_state=t1.select(t1.root,state);
                if (expand_state!=None):

                    batch=torch.stack([expand_state.getNNinput()]).to("cuda");
                    policies,values=model1(batch);
                    policies=policies.to("cpu").detach().numpy();
                    values=values.to("cpu").detach().numpy();
                    t1.expand(node,expand_state,policies[0],values[0][0]);
                    t1.propogate(node,values[0][0]);
                
                node,expand_state=t2.select(t2.root,state);
                if (expand_state!=None):

                    batch=torch.stack([expand_state.getNNinput()]).to("cuda");
                    policies,values=model2(batch);
                    policies=policies.to("cpu").detach().numpy();
                    values=values.to("cpu").detach().numpy();
                    t2.expand(node,expand_state,policies[0],values[0][0]);
                    t2.propogate(node,values[0][0]);
            


            tree=players[idx];
            nextNodes=[tree.root.children[k] for k in range(len(tree.root.children))]

            maxSim=max([node.N for node in nextNodes])
            allMaxSim=[]

            for node in nextNodes:
                if (node.N==maxSim):
                    allMaxSim.append(node)
            nextRoot=random.choice(allMaxSim)
            state=state.nextState(nextRoot.actionDone,'X' if tree.root.player else 'O');
            tree.root=nextRoot
            tree.root.parent=None;
            
            
            tree=players[idx^1];
            otherRoot=None;
            for node in tree.root.children:
                if (node.actionDone==nextRoot.actionDone):
                    otherRoot=node;
                    break 
            tree.root=otherRoot
            tree.root.parent=None; 
            idx^=1
            state.display();
            print("\n")
        winner=state.checkWinner();
        if (winner==1):
            if (swapped):
                win+=1 
            else:
                loss+=1 
        elif (winner==-1):
            if (swapped):
                loss+=1 
            else:
                win+=1 
        else:
            draw+=1 
        
        print("Test Game: " + str(i+1) + " Done")
    print("Win Rate: " + str((win)/(win+draw+loss)*100))
    print("Loss Rate:" + str((loss)/(win+loss+draw)*100))
    print("Draw Rate: " + str(draw/(draw+win+loss)*100));

    if ((win+draw)/(win+draw+loss)<THRESHOLD):
        
        return False 
    print("Model Replaced");
    return True 

if __name__=="__main__":
    
    PTH="/home/jay/Desktop/AlphaUTTT(Torch)/model.pt"
    checkpoint = torch.load(PTH);
    model=Net();
    model.load_state_dict(checkpoint['model_state_dict']);
    model.to(device="cuda");
    optimizer=torch.optim.Adam(model.parameters());
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']);      
    model.eval()
    model=torch_tensorrt.compile(model,inputs=[torch_tensorrt.Input(min_shape=[1,2,9,9],opt_shape=[1,2,9,9],max_shape=[1,2,9,9])],enabled_precisions={torch.float})

    battle(model,model)