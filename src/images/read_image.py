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


def read(filepath: Union[List[Path],Path],resize: int = None):
    if isinstance(filepath,list) or filepath.is_dir():
        array = np.array([])
        for file in filepath:
            image = read(filepath=file,resize=resize)
            array = np.append(array,image)
        return array
    else:
            image = cv.imread(filename=str(filepath))
            cv.cvtColor(src=image,code=cv.COLOR_BGR2GRAY,dst=image)
            if resize is not None and resize > 0:
                cv.resize(src=image,dsize=(resize,resize),dst=image)
            cv.equalizeHist(src=image,dst=image)
            return image