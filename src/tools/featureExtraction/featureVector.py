import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from PIL import Image as PILImage

import torchvision.models as models
from torchvision import transforms

from common.image.Image import Image


def get_feat_vector(img: [Image], model: models) -> Tensor:
    """
    function to get the feature at the avgpool
    Input:
        img: Image, image object
        model: a pretrained torch model
    return: Tensor, output of avgpool layer
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Process img with the transforms then unsqueeze it to be able to feed it to the model
    # Todo: faire le code pour avoir plusieurs images
    input_batch = preprocess(PILImage.fromarray(np.asarray(img[0].value))).unsqueeze(0)

    # Construct a new model by removing layers after avgpool
    new_model = nn.Sequential(*list(model.children())[:-2])
    with torch.no_grad():
        model = new_model(input_batch)
        print(model)
        return model
