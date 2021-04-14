from src.images.read_image import read
from typing import Any, List, Tuple
from keras.utils.data_utils import Sequence
import numpy as np

class VolumeGenerator(Sequence):

    def __init__(
        self,
        data: Tuple[str,str,str],
        dimension: int = 512,
        batch_size: int = 32
    ) -> None:
        super(VolumeGenerator, self).__init__()
        self.x1, self.x2, self.y = data
        self.batch_size = batch_size
        self.dim = dimension
    
    def __len__(self) -> int:
        return int(np.floor(self.x1) / self.batch_size)

    def __iter__(self) -> int:
        self.n = 0
        return self

    def __getitem__(
        self,
        index: int
    ) -> Tuple[Any, List[int], List[float]]:
        batch_x1 = self.x1[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x2 = self.x2[index * self.batch_size: (index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size: (index + 1) * self.batch_size]

        batch_x1 = read(batch_x1, resize=self.dimension)
        batch_x2 = [int(value) for value in batch_x2]
        batch_y = [float(value) for value in  batch_y]

        return batch_x1, batch_x2, batch_y