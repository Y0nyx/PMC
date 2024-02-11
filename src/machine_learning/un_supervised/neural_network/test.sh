#!/bin/bash
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 02/07/2024
Description: Run test_AE on a super_computer or localy. 
"""
readonly PATH_NN='/home/jean-sebastien/Documents/s7/PMC/PMC/src/machine_learning/un_supervised/neural_network'
readonly PATH_VE='/home/jean-sebastien/Documents/s7/PMC/venv/bin/activate'
readonly SCRIPT_NAME='test.py'

readonly DATA_PATH='/home/jean-sebastien/Documents/s7/PMC/Data/images_cam_123/sub_images' #same as test_AE.sh
readonly MODEL='build_basic_cae'     #same as test_AE.sh
readonly TEST_NAME='First_HP_Search' #same as test_AE.sh
readonly NBEST=5                     #same as test_AE.sh
readonly NUM_TRAIN_REGENERATE=20

#Never change this constants (Only adapt PATH_RESULTS to the computer you are using)
readonly PATH_RESULTS="/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/${MODEL}/${TEST_NAME}"
readonly FILEPATH_WEIGHTS="${PATH_RESULTS}/training_weights/"


#Activate VE
if [ -f "$PATH_VE" ]; then
    source "$PATH_VE"
    echo 'Success to open the venv.'
else
    echo 'Failed to activate the virtual environment.'
    exit 1
fi

#Run the script test_AE.py
cd $PATH_NN
./$SCRIPT_NAME --PATH_RESULTS $PATH_RESULTS --NBEST $NBEST --NUM_TRAIN_REGENERATE $NUM_TRAIN_REGENERATE --FILEPATH_WEIGHTS $FILEPATH_WEIGHTS --DATA_PATH $DATA_PATH