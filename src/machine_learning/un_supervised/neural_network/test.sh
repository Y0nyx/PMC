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

readonly DATA_PATH='/home/jean-sebastien/Documents/s7/PMC/Data/Datasets_segmentation_grayscale' #/home/jean-sebastien/Documents/s7/PMC/Data/4k_dataset/original_images/sans_defauts_full_data_blanc #'../../../../../Datasets/grosse_piece_seg_1'   #TODO convert for docker     #same as train.sh
readonly MODEL='aes_defect_detection'                                                     #same as train.sh
readonly TEST_NAME='pmc860_grayscale_dataset'                                             #same as train.sh
readonly NBEST=1                                                                          #same as train.sh
readonly NUM_TRAIN_REGENERATE=300
readonly MONITOR_LOSS='mean_absolute_error'
readonly MONITOR_METRIC='mean_squared_error'
readonly MAX_PIXEL_VALUE=255
readonly SUB_WIDTH=256
readonly SUB_HEIGHT=256

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
./$SCRIPT_NAME --PATH_RESULTS $PATH_RESULTS --NBEST $NBEST --NUM_TRAIN_REGENERATE $NUM_TRAIN_REGENERATE --MONITOR_LOSS $MONITOR_LOSS --MONITOR_METRIC $MONITOR_METRIC --FILEPATH_WEIGHTS $FILEPATH_WEIGHTS --DATA_PATH $DATA_PATH --MAX_PIXEL_VALUE $MAX_PIXEL_VALUE --SUB_WIDTH $SUB_WIDTH --SUB_HEIGHT $SUB_HEIGHT