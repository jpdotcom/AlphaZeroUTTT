from mcts import MCTS
#import tensorflow as tf
#from tensorflow import keras 
from UTTT import UTTT 
import random;
import torch
from torchModel import Net
import numpy as np;
from C4 import C4
import matplotlib.pyplot as plt
import torch_tensorrt
np.set_printoptions(suppress=True)
state=UTTT([0]*10,None);
#state=C4([0,0]);
PTH="/home/jay/Desktop/AlphaUTTT(Torch)/model.pt"
checkpoint = torch.load(PTH);
model=Net();
model.load_state_dict(checkpoint['model_state_dict']);
model.to(device="cuda");
optimizer=torch.optim.Adam(model.parameters());
optimizer.load_state_dict(checkpoint['optimizer_state_dict']);
tree=MCTS(False);
player=0;
model.eval()
rtModel=torch_tensorrt.compile(model,inputs=[torch_tensorrt.Input(min_shape=[1,2,9,9],opt_shape=[1,2,9,9],max_shape=[1,2,9,9])],enabled_precisions={torch.float})

SIM=5000;
while (not state.gameOver()):
    

    if (player):
        print(f"Human valid moves: {state.getValidMoves()}")
        actionDone=int(input());
        
        state=state.nextState(actionDone,"X");
    
        for node in tree.root.children:
            if (node.actionDone==actionDone):
                tree.root=node;
                break; 
        print("\n")
    else:

        
        for j in range(SIM):
            #node,value=t1.select();
            #t1.propogate(node.state,node,value);

            #node,value=t2.select();
            #t2.propogate(node.state,node,value);
            node,expand_state=tree.select(tree.root,state);
            if (expand_state!=None):
                nnInput=expand_state.getNNinput();
                nnInput[0]=(nnInput[0]*-1 if node.player else nnInput[0])
                batch=torch.stack([nnInput]).to("cuda");
                policies,values=rtModel(batch);
                policies=policies.to("cpu").detach().numpy();
                values=values.to("cpu").detach().numpy();
                tree.expand(node,expand_state,policies[0],values[0][0]);
                tree.propogate(node,values[0][0]);
        nextNodes=[tree.root.children[k] for k in range(len(tree.root.children))]
       
        nninput=[state.getNNinput()]
        print(nninput[0])
        nninput=torch.stack(nninput);
        nninput=nninput.to("cuda")
        
        priors,values=model(nninput)

        priors=priors.to("cpu");
        values=values.to("cpu");
        priors=priors.detach().numpy();
        priors=priors[0]
        
        priors=np.exp(priors);
        
        priors=priors/np.sum(priors);
        print(f"Bot valid moves: {state.getValidMoves()}")
        fig = plt.figure(figsize = (10, 5))
        plt.bar([i for i in range(0,81)], priors, width=0.4, color="maroon");
        plt.show()
        #print(f"Network Policy values: {priors}")
   
        print(f"Network value: {values.detach().numpy()[0][0]}. MCTS Q_value: {tree.root.Q}" );
    
        
        #print(priors);
        #print("Q_value: " + str(float(tree.root.Q)));
        #print("Value: " + str(values.detach().numpy()[0][0]));
        #print(tree.root.v);
        maxSim=max([node.N for node in nextNodes])
        print(f"Sim count: {[node.N for node in nextNodes]}")
        print(f"MCTS Policy values: {np.array([node.N for node in nextNodes])/np.sum([node.N for node in nextNodes])}")
        allMaxSim=[]

        for node in nextNodes:
            if (node.N==maxSim):
                allMaxSim.append(node)
        nextRoot=random.choice(allMaxSim)
        state=state.nextState(nextRoot.actionDone,'O');
        tree.root=nextRoot;
    state.display();
    #values=float(predict(model,state.getNNinput())[1]);
    #print(values);
    player^=1;
    print("\n")
