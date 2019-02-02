
#All local file imports
#------------------------------------------------------------------------
import constant
import dataset
import test
import train
import model
import visualize
#=========================================================================


#All basic imports 
#-----------------------------------------------------------------------------
import numpy as np
import os
import sys
sys.path.append(constant.MRCNN_DIR)
#=============================================================================


#Everything mrcnn goes here
#---------------------------------------------------------------------------
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
#=============================================================================

if __name__=="__main__":

	dataset_train=
	dataset_val=

	config = networkConfig()
	model = modellib.MaskRCNN(mode="training", config=config, \
							model_dir=constant.MODEL_DIR)

	#Train head layers
	model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, 
            epochs=100, layers='heads')

	#Finetune all layers
	model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10,
            epochs=100, layers="all")

