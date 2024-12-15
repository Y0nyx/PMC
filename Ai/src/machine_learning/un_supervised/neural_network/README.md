         For classification, the trained network will process data containing both defect-free and defective pieces. 
         Once the network infers a reconstructed image of a welding piece, we will analyze loss metrics such as RMSE, MAE, or SSIM 
         by comparing the original image with the network's prediction.

         We expect the loss metrics to be higher for pieces containing defects and lower for defect-free pieces, as the network is trained 
         to reconstruct healthy samples. Based on a predefined threshold, we will determine whether a piece contains a defect or not.

         This workflow leverages the network's ability to distinguish between healthy and defective welding pieces by analyzing its reconstruction performance.



        train.sh : Bash file to select variables and constants, create virtual environment and lanch train.py
        train.py: Main file to train and do a hyper-parameter search for a selected model
        data_processing.py: Class where the data processing is done (Ideally, data processing is done only once for one dataset (Save data after this to save time and resources))
        callbacks.py: Class of callbacks used in the training of the neural network to help training if certain conditions are met.
        hyper_parameters_tuner.py: Class to do the hyper-parameter search with Keras-Tuner.
        model.py: Class where the neural networks models are defined. 
        training_info.py: Wehere graph during training are done and saved. 
        test.sh: Bash file to select variables and constants, create virtual environment and lanch test.py.
        test.py: Do the Neural Network inference on the test data and classify the welding results. 