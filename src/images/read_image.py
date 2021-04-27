from os import error
from src.data.convertion import np2cv
from typing import Any, List, Tuple, Union
import nibabel as nib
from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage

def read_nii(filepath: Union[Path, List[Path]]):
    if isinstance(filepath, list):
        array = np.array([read_nii(filepath=file) for file in filepath])
        return array
    if filepath.suffix == '.nii':
        ct = nib.load(filepath)
        ct = ct.get_fdata()
        ct = np.rot90(np.array(ct))
        return ct

def normalize_volume(volume: Any):
    """Normalize the volume
    
    Args:
        volume (np.array): Vetor de imagens contento as imagens CTs.
    Return:
        (np.array): CT normalizado
    """
    min = 0
    max = 255

    volume[volume < min] = min
    volume[volume > max] = max

    volume = (volume - min) / (max - min)
    volume = volume.astype('float32')
    
    return volume

def resize_volume(
    image:Any,
    depth: int = 325,
    height: int = 512,
    width: int = 512,
    shape: Union[Tuple[int,int,int],None] = None
) -> Any:
    """Resize across z-axis"""
    # Set the desired depth
    desired_height = shape[0] if shape is not None else height
    desired_width = shape[1] if shape is not None else width
    desired_depth = shape[-1] if shape is not None else depth

    # Get current depth
    current_height = image.shape[1]
    current_width = image.shape[-1]
    current_depth = image.shape[0]

    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Rotate
    img = ndimage.rotate(
        input=image,
        angle=90,
        reshape=False
    )
    img = ndimage.zoom(
        input=image,
        zoom=(height_factor,width_factor,depth_factor),
        order=1
    )
    return img

def process_scan(path: Path):
    """Read and resize volume"""
    # Read scan
    volume = read_nii(filepath=path)
    volume = normalize_volume(volume=volume)
    volume = resize_volume(image=volume)
    return volume

def show_slices(nii_file: Path) -> None:
    """Function to display row of image slices"""
    slices = process_scan(nii_file)
    ax = plt.axes(projection='3d')
    shape = slices.shape
    choices = range(
        0,
        shape[0],
        int(shape[0] / 10)
    )
    slices = slices[choices,:,:]
    zline, yline, xline, cline = [], [], [], []
    for i, slice in enumerate(slices):
        args = np.argwhere(slice > 0.7)
        if args.size > 0:
            y, x = args[:,0], args[:,1]
            z = np.array([i] * len(x))
            c = np.array([1] * len(x))
            zline = np.append(zline,z)
            yline = np.append(yline,y)
            xline = np.append(xline,x)
            cline = np.append(cline,c)
    ax.scatter3D(xline, yline, zline,s=1)
    plt.show()

def read(
    filepath: Union[List[Path], Path],
    resize: int = None
):
    if isinstance(filepath, list) or filepath.is_dir():
        array = np.array([])
        for file in filepath:
            image = read(filepath=file, resize=resize)
            array = np.append(array, image)
        return array
    else:
        image = cv.imread(filename=str(filepath))
        image_gray = cv.cvtColor(src=image, code=cv.COLOR_BGR2GRAY, dst=image)
        if resize is not None and resize > 0:
            image_resize = cv.resize(
                src=image_gray,
                dsize=(resize, resize),
                interpolation=cv.INTER_AREA,
            )
            image_gray = image_resize
        image = image_gray / 256
        return image
