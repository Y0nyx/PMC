#!/bin/bash
"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/25/2024
Description: Training Neural Network.
"""

#Definition of read only variables (Constants)
#Variable for environment set-up
readonly PATH_NN='/home/jean-sebastien/Documents/s7/PMC/PMC/src/machine_learning/un_supervised/neural_network'
readonly PATH_VE='/home/jean-sebastien/Documents/s7/PMC/venv/bin/activate'
readonly SCRIPT_NAME='train.py'

#Variable for Callbacks
readonly MODEL='ae_fchollet'
readonly TEST_NAME='First_HP_Search'
readonly MONITOR_METRIC='mean_absolute_error'
readonly MODE_METRIC='min'
readonly VERBOSE=TRUE #If false remove --VERBOSE from line 40
#Variable for HP search 
readonly DO_HP_SEARCH=True #If false remove --DO_HP_SEARCH from line 40
readonly EPOCHS_HP=3
readonly NUM_TRIALS_HP=3
readonly EXECUTION_PER_TRIAL_HP=1
readonly NBEST=2
#Variable for default training
readonly EPOCHS=5
readonly BATCH_SIZE=256

#Never change this constants (Only adapt PATH_RESULTS to the computer you are using)
readonly PATH_RESULTS="/home/jean-sebastien/Documents/s7/PMC/results_un_supervised/${MODEL}/${TEST_NAME}"
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
./$SCRIPT_NAME --EPOCHS $EPOCHS --BATCH_SIZE $BATCH_SIZE --PATH_RESULTS $PATH_RESULTS --FILEPATH_WEIGHTS $FILEPATH_WEIGHTS --FILEPATH_WEIGHTS_SERCH $FILEPATH_WEIGHTS_SERCH --HP_SEARCH $HP_SEARCH --HP_NAME $HP_NAME --MONITOR_METRIC $MONITOR_METRIC --MODE_METRIC $MODE_METRIC --VERBOSE --DO_HP_SEARCH --EPOCHS_HP $EPOCHS_HP --NUM_TRIALS_HP $NUM_TRIALS_HP --EXECUTION_PER_TRIAL_HP $EXECUTION_PER_TRIAL_HP --NBEST $NBEST
