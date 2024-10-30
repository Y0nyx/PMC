#!/bin/bash
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Training Neural Network.
"""

#Definition of read only variables (Constants)
#Variable for environment set-up
readonly PATH_NN='/home/jean-sebastien/Documents/s7/PMC/PMC/src/machine_learning/un_supervised/neural_network'  #/PMC/src/machine_learning/un_supervised/neural_network
readonly PATH_VE='/home/jean-sebastien/Documents/s7/PMC/venv/bin/activate' #Utiliser doker
readonly SCRIPT_NAME='train.py'

readonly DATA_PATH='/home/jean-sebastien/Documents/s7/PMC/Data/Datasets_segmentation_grayscale' #'/home/jean-sebastien/Documents/s7/PMC/Data/4k_dataset/original_images/sans_defauts_full_data_blanc' In PMC 760 that was the datapath
#Variable for Callbacks
readonly MODEL='aes_defect_detection'
readonly TEST_NAME='pmc860_grayscale_dataset_0'
readonly MONITOR_LOSS='mean_absolute_error'
readonly MONITOR_METRIC='mean_squared_error'
readonly MODE_METRIC='min'
readonly VERBOSE=True #If false remove --VERBOSE from line 40
#Variable for HP search 
readonly DO_HP_SEARCH=True #If false remove --DO_HP_SEARCH from line 40
readonly EPOCHS_HP=40
readonly NUM_TRIALS_HP=5
readonly EXECUTION_PER_TRIAL_HP=1
readonly NBEST=1
#Variable for default training
readonly EPOCHS=5
readonly BATCH_SIZE=256
readonly LEARNING_RATE=0.001
#Variable for image information 
readonly MAX_PIXEL_VALUE=255
readonly SUB_WIDTH=256
readonly SUB_HEIGHT=256

#Never change this constants (Only adapt PATH_RESULTS to the computer you are using)
readonly PATH_RESULTS="/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/${MODEL}/${TEST_NAME}" #"/un_supervised_training_results/${MODEL}/${TEST_NAME}" #TODO convert for docker
readonly FILEPATH_WEIGHTS="${PATH_RESULTS}/training_weights/"
readonly FILEPATH_WEIGHTS_SERCH="${PATH_RESULTS}/search_weights/"
readonly HP_SEARCH="${PATH_RESULTS}/"
readonly HP_NAME="hp_search_results"

#Activate VE
if [ -f "$PATH_VE" ]; then
    source "$PATH_VE"
    echo 'Success to open the venv.'
else
    echo 'Failed to activate the virtual environment.'
    exit 1
fi

#Run the script train_AE.py
cd $PATH_NN
./$SCRIPT_NAME --EPOCHS $EPOCHS --BATCH_SIZE $BATCH_SIZE --LEARNING_RATE $LEARNING_RATE --PATH_RESULTS $PATH_RESULTS --FILEPATH_WEIGHTS $FILEPATH_WEIGHTS --FILEPATH_WEIGHTS_SERCH $FILEPATH_WEIGHTS_SERCH --HP_SEARCH $HP_SEARCH --HP_NAME $HP_NAME --MONITOR_LOSS $MONITOR_LOSS --MONITOR_METRIC $MONITOR_METRIC --MODE_METRIC $MODE_METRIC --VERBOSE --DO_HP_SEARCH --EPOCHS_HP $EPOCHS_HP --NUM_TRIALS_HP $NUM_TRIALS_HP --EXECUTION_PER_TRIAL_HP $EXECUTION_PER_TRIAL_HP --NBEST $NBEST --DATA_PATH $DATA_PATH --MAX_PIXEL_VALUE $MAX_PIXEL_VALUE --SUB_WIDTH $SUB_WIDTH --SUB_HEIGHT $SUB_HEIGHT
