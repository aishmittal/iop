#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:06:06 2017
TEST SCRIPT
@author: shubham
"""

import database_operations as dsop
import cnn_model_generator as cmg

#MY GLOBAL CONSTANTS
LOAD_PATH='/home/aishwarya/Documents/IOP/data/images/att_faces'#load data from here
STORE_PATH='/home/aishwarya/Documents/IOP/data'#store data in this folder
DSETP=STORE_PATH+'/datasets'
MODLP=STORE_PATH+'/models'

DSETFN="att_faces"
MFN="model_"+DSETFN# model filename

NEPOCH=800
DBS=500# data batch size
TBS=10# training batch size
DMY="no val"# dummmy

dsop.createSingleBlockDataset(LOAD_PATH,DSETP,DSETFN,(50,50,3))
md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_metadata.txt')
#print(md[0],md[0]["shape"])
#
#print("PATH:",DSETP+"/"+DSETFN+".h5",md[0]["shape"])
dsop.navigateDataset(DSETP+"/"+DSETFN+".h5",md[0]["shape"],0)

dsop.partitionDataset(DSETP+"/"+DSETFN+".h5",DSETP+"/"+DSETFN+"_metadata.txt",(60,40))

md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_train_metadata.txt')
model= cmg.getModelFrame(md[0]["shape"],int(md[0]["nb_classes"]))
DBS= md[0]["dataset_shape"][0]
MFN=MFN+"_"+str(NEPOCH)
model_path=cmg.getTrainedModel(model,DSETP+"/"+DSETFN+"_train.h5",DSETP+"/"+DSETFN+"_train_metadata.txt",
                              MODLP,MFN,NEPOCH,DBS,TBS)
print(model_path)



MODEL_LOC=MODLP+"/"+MFN+".h5"
TD_LOC=DSETP+"/"+DSETFN+"_test.h5"
TD_MD_LOC=DSETP+"/"+DSETFN+"_test_metadata.txt"
cmg.evaluateModel (MODEL_LOC,TD_LOC,TD_MD_LOC)











        
