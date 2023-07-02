import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from PIL import Image as PILImage

import torchvision.models as models
from torchvision import transforms

from src.common.image.Image import Image


def get_feat_vector(img: Image, model: models) -> Tensor:
    '''
    Input:
        img: Image, image object
        model: a pretrained torch model
    Output:
        output: torch.tensor, output of avgpool layer
    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(PILImage.fromarray(np.asarray(img.value)))
    input_batch = input_tensor.unsqueeze(0)

    # Construct a new model by removing layers after avgpool
    new_model = nn.Sequential(*list(model.children())[:-2])

    with torch.no_grad():
        output = new_model(input_batch)

    return output

