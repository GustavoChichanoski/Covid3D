import numpy as np

def calc_lung_area_px(mask) -> int:
    mask = (mask > 0.8).astype(np.float32)
    return np.sum(mask)

def segmentation_lung(lung,mask):
    return lung * mask