import pytest
from unittest.mock import MagicMock, path, mock_open
import os
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Doing the imports for the tested file:
import os
import warnings
import numpy as np
from pathlib import Path
from common.image.Image import Image
from common.image.ImageCollection import ImageCollection
from common.image import utils

@pytest.fixture
def mock_dataset(tmpdir):
    """
    Set up a mock dataset with temporary directories and image files. 
    """
    # Create a temporary directory structure
    dataset_path = tmpdir.mkdir("dataset")
    iteration_dirs = dataset_path.mkdir("iteration_01")

    # Create some mock image file in the directory
    num_img = 3
    for i in range(num_img):
        img_file = iteration_dirs.join(f"image_{i}.jpg")
        img_file.write("fake image content")
    
    return dataset_path

@pytest.fixture
def data_manager(mock_dataset):
    """
    Fixture create and returns a DataManager instance with the mock dataset
    """
    return utils.DataManager(str(mock_dataset))


# def test_