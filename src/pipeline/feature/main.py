import urllib
import numpy as np
from PIL import Image
from FeatureExtractionResnet import FeatureExtractionResnet

from src.common.image.Image import Image as Img

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

if __name__ == '__main__':
    featureExtraction = FeatureExtractionResnet()

    input_image = Img(np.asarray(Image.open(filename)))

    features = featureExtraction.get_feature(input_image)

    if features is None:
        print('Failed to extract features')
    else:
        print('Extracted features successfully')


