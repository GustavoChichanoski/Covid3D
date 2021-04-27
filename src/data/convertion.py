from typing import Tuple
import numpy as np
import cv2 as cv


def np2cv(value):
    return value.astype(np.uint8)


def np2keras(value, shape: Tuple[int, int, int]):
    keras = np.array(value)
    return keras.reshape(shape)