import tensorflow as tf
from tensorflow import keras
import random
import numpy as np;
import queue

BATCH_SIZE=200;




def create_model(weights=None):

    # input_layer=keras.layers.Input(shape=(3,3,11))
    # backbone=keras.Sequential([
    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),

    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),
        
    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),
        
    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),
        
    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),

    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Conv2D(700,(2,2),activation="relu"),
        
        
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(2048,activation="relu")
        
    # ]



    # )
    

    # policyNet=keras.models.Sequential([
    #     keras.layers.Dense(81,input_shape=(2048,),activation="softmax")
    # ])

    # valueNet=keras.models.Sequential([
    #     keras.layers.Dense(1,input_shape=(2048,),activation="tanh")
    # ])

    input_layer=keras.layers.Input(shape=(3,3,1))
    backbone=keras.Sequential([
        # keras.layers.Conv2D(700,(2,2),activation="relu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),

        # keras.layers.Conv2D(700,(2,2),activation="relu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),
        
        # keras.layers.Conv2D(700,(2,2),activation="relu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),
        
        # keras.layers.Conv2D(700,(2,2),activation="relu"),
        # keras.layers.BatchNormalization(),
        # keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),
        
        keras.layers.Conv2D(700,(2,2),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(700,(2,2),activation="relu"),

        keras.layers.Conv2D(700,(2,2),activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(700,(2,2),activation="relu"),
        
        
        keras.layers.Flatten(),
        keras.layers.Dense(300,activation="relu")
        
    ]



    )
    

    policyNet=keras.models.Sequential([
        keras.layers.Dense(9,input_shape=(300,),activation="softmax")
    ])

    valueNet=keras.models.Sequential([
        keras.layers.Dense(1,input_shape=(300,),activation="tanh")
    ])




    latent=backbone(input_layer);

    output1=policyNet(latent);
    output2=valueNet(latent);

    model=keras.models.Model(inputs=input_layer,outputs=[output1,output2]);
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.Adam(0.001))
    if (weights!=None):
        model.set_weights(weights);
    return model

#print(create_model().summary());
def train(model,examples):

    indices=[random.randint(0,len(examples)-1) for i in range(BATCH_SIZE)]

    batch=[examples[i] for i in range(len(examples))]
    random.shuffle(batch);
    x_train=[batch[i][0] for i in range(len(batch))]
    py_train=np.array([batch[i][1] for i in range(len(batch))])
    vy_train=np.array([batch[i][2] for i in range(len(batch))])
    x_train=np.array(x_train);
    x_train=np.transpose(x_train,(0,2,3,1))
    #print(np.sum(vy_train))
    model.fit(x_train,[py_train,vy_train],BATCH_SIZE,epochs=30)
    
@tf.function
def predict(model,s):
  
    
   
    s=np.transpose(s,(1,2,0))
    s=s.reshape(1,s.shape[0],s.shape[1],s.shape[2])
    s=tf.convert_to_tensor(s,dtype=tf.float32);
    out = model(s,training=False);
  
    return out;