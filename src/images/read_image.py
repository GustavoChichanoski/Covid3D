from typing import List, Union
import nibabel as nib
from pathlib import Path
import numpy as np
import cv2 as cv


def read_nii(filepath: Union[Path, List[Path]]):
    if isinstance(filepath, list):
        array = np.array([read_nii(filepath=file) for file in filepath])
        return array
    if filepath.suffix == '.nii':
        ct_scan = nib.load(filepath)
        array = ct_scan.get_fdata()
        array = np.rot90(np.array(array))
        return array


def read(filepath: Union[List[Path],Path]):
    if isinstance(filepath,list):
        array = np.array([])
        for file in filepath:
            image = cv.imread(str(file))
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            array = np.append(array,image)
        return array
    else:
            image = cv.imread(str(filepath))
            return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
