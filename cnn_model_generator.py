#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 00:13:59 2017

@author: shubham
"""
#MY IMPORTS
import database_operations as loader


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import load_model
from keras.utils import np_utils
from keras import backend as K

K.set_epsilon(1e-6)


def getModelFrame (img_shape,nb_classes):
    assert(len(img_shape)==3)
    input_img = Input(shape=(img_shape[2],img_shape[0],img_shape[1]))
    
    c = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)#1
    c = MaxPooling2D((2, 2), border_mode='same')(c)#16*64*64
    
    c = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(c)#2
    c = MaxPooling2D((2, 2), border_mode='same')(c)#8*32*32
    
    c = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(c)#2
    c = MaxPooling2D((2, 2), border_mode='same')(c)#8*32*32                
    
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)#3
    c= MaxPooling2D((2, 2), border_mode='same')(c)#8*16*16
    
    c= Convolution2D(128,3,3,activation='relu',border_mode='same')(c)#5
    joint_layer=MaxPooling2D((2,2),border_mode='same')(c)#8*4*4
    
    FCN_layer= Flatten()(joint_layer)#d1
    d=Dense(100, activation='sigmoid')(FCN_layer)#d2
    d=Dropout(0.5)(d)
    
    
    d=Dense(80, activation='sigmoid')(d)#d3
    d=Dropout(0.5)(d)
    
    
    d=Dense(40, activation='sigmoid')(d)#d3
    d=Dropout(0.5)(d)
    #op_layer_num_neurons=input('Enter op layer neuron number: ')
    output=Dense(nb_classes,activation='softmax')(d)
    model=Model(input_img,output,name="model")
    print("type", type(model))
    
    #optimizer_fn= SGD(nesterov=True)
    optimizer_fn=RMSprop()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_fn,
                  metrics=['accuracy'])
    return model

def getTrainedModel(model_to_train,dataset_loc,metadata_loc,
                    STORE_FOLDER, op_model_file_name,num_of_epoch=111,data_batch_size=100,training_batch_size=10,
                    create_metadata=True):             
    """
    dataset_loc: full paths required e.g. /home/user/dtop/test/file.h5
    same for metadata_loc
    "add kwargs: for md path"
    """
    
    
    with loader.hf.File(dataset_loc,'r') as f:
        num_epoch=num_of_epoch
        metadata=loader.loadMetaData(metadata_loc)
        nb_classes= int(metadata[0]["nb_classes"])
        next_batch=loader.getNextBatch
        #b_size=metadata[0]["dataset_shape"][0]
        b_size=data_batch_size
        print("Batch_size:",b_size,"metadata[0]=",metadata[0])
        #wait=input("Enter 1 and proceed;")
        y=loader.np.zeros(shape=(b_size,nb_classes))
       
        x,y_tr=next_batch(b_size,f,metadata)
        x=x.astype('float32')
        x=x.transpose(0,3,1,2)
        x=(x-100)/255
        print("x:",x.shape)
        print("y:",y.shape)
        print(y_tr)
        #for j in range(len(y_tr)):
            #print()
            #y[j][y_tr[j][0]]=1
        y = np_utils.to_categorical(y_tr, nb_classes)
        print("Y:",y)
        model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch)
        generated_model_address=STORE_FOLDER+'/'+op_model_file_name+'.h5'
        model_to_train.save(generated_model_address)
        
        
        if create_metadata:
            specs=metadata[0]
            specs.update({"original_model_address":generated_model_address,
                   "dataset_trained_on":dataset_loc,
                   "num_of_epoch":num_of_epoch,
                   "data_batch_size":data_batch_size,
                   "training_batch_size":training_batch_size
                   })
            loader.generateMetaData(STORE_FOLDER+'/'+op_model_file_name+'_metadata.txt',
                                    specs,metadata[1])
        
        
        
        return generated_model_address
    
    
def evaluateModel (model_loc,test_dataset_loc,test_metadata_loc,use_whole_dataset=True,percentage_used=100):
    with loader.hf.File(test_dataset_loc,'r') as f:
         md=loader.loadMetaData(test_metadata_loc)
         
         if use_whole_dataset:    
             x,y=loader.getNextBatch(md[0]["dataset_shape"][0],f,md)
         else:
             assert(percentage_used<=100)
             num_samples= int((md[0]["dataset_shape"][0])*percentage_used*0.01)
             x,y=loader.getNextBatch(num_samples,f,test_metadata_loc)

         x_test=x.transpose(0,3,1,2)
         x_test=x_test.astype('float32')
         x_test=(x_test-100)/255
         print("X_Test.dtype",x_test.dtype)
         model= load_model(model_loc)
         pred_label=model.predict(x_test)
         wrong_count=0;
         for i in range(x.shape[0]):
            #cv.putText(x[i],str(y[i][0]),(60,90), cv.FONT_HERSHEY_COMPLEX, 1,(0,0,255),1)
            #cv.rectangle(x[i],(55,85),(100,100),(0,255,0),3)
            
            #print(pred_label.shape)
            #for itr in range (pred_label.shape[0]):
            pr=loader.np.argmax(pred_label[i])
            print("itr:",i," Pred:",pr)
            print("Actual_LAbel:",y[i][0])
            if pr==y[i][0]:
                pass
            else:
                wrong_count+=1
         print ("Wrong:",wrong_count," %age Acc:", (1-wrong_count/x.shape[0])*100)

             

