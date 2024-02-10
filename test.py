import math;
import numpy as np;
import random;
import time;
from TicTacToe import *
from model import *
from UTTT import UTTT;
from threading import Thread
from threading import Lock;
from collections import deque
import json
import multiprocessing
from multiprocessing import Process
#import tensorflow as tf
from pit import battle;
from mcts import MCTS
import gc
with open('examples.json','r') as file:

    loadExamples=json.load(file);
    print(len(loadExamples))
#loadExamples=[]
#THREAD_COUNT = 5
#results=[None]*THREAD_COUNT;

allExamples=deque(loadExamples,maxlen=10000)
loadExamples=[]
mainModel=keras.models.load_model("/Users/jaypatel/Desktop/AlphaUTTT/Model")
model=create_model();
model.set_weights(mainModel.get_weights());
i=0;
while (True):
    
    idx=random.randint(0,len(allExamples)-1);
    s=allExamples[i][0];
   
    v1,v2=predict(model,s).numpy();