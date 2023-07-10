from NNManager import NNManager
#from common.image.Image import Image
#from pipeline.data.dataManager import DataManger
from pipeline.feature.featureExtraction import FeatureExtraction
from pipeline.feature.featureExtractionManager import FeatureExtractionManager
from torch import Tensor

print("Here we go!")
# Create an instance of NNManager
nn_manager = NNManager.get_instance()

# Create an instance of FeatureExtractionManager
feature_extraction_manager = FeatureExtractionManager()

# Set the feature extractions in FeatureExtractionManager
feature_extractions = [FeatureExtraction(), FeatureExtraction()]  # Replace with actual feature extraction instances
feature_extraction_manager._featureExtractions = feature_extractions

# Retrieve all features from FeatureExtractionManager
all_features = feature_extraction_manager.get_all_features()

# Retrieve feature from a specific index
index = 0  # Replace with a valid index
feature = feature_extraction_manager.get_feature(index)

# Retrieve the state of all feature extractions
states = feature_extraction_manager.get_state()

# Print the results
print("All Features:", all_features)
print("Feature at index", index, ":", feature)
print("Feature Extraction States:", states)