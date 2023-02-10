import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    files = []
    for i in os.walk(dir_name):
      for j in i[2]:
        if j.__contains__('.'):
          files.append(os.path.join(i[0],j))

    image_rgbs = []
    for i in files:
      image_rgbs.extend(list(np.array(Image.open(i)).flatten()/255))
    image_rgbs = np.array(image_rgbs)
    mean = np.mean(image_rgbs)
    std = np.std(image_rgbs)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
