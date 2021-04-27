from pathlib import Path
from src.images.read_image import read
from typing import Any, List, Tuple, Union
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np


class Generator(Sequence):

    def __init__(
        self,
        data: Tuple[str, str],
        batch_size: int = 32,
        dim: int = 512,
        n_class: int = 1,
        n_channel: int = 1
    ) -> None:
        super(Generator, self).__init__()
        self.x, self.y = data
        self.batch_size = batch_size
        self.dim = dim
        self.n_class = n_class
        self.n_channels = n_channel

    def __len__(self) -> int:
        """
            Retorna o numero de passos do conjunto

            Returns:
            --------
                int: numero de passos do dataset
        """
        return int(np.floor(len(self.x) / self.batch_size))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.__len__(self):
            result = self.__getitem__(self, self.n)
            self.n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
            Retorna os valores a serem lidos em cada passo do
            treinamento. IDx o índice inicial e retorna o batch
            do passo.

            Args:
            -----
                idx (int): Posicao inicial

            Returns:
            --------
                tuple(np.array, np.array): pacotes X e Y
        """
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x = read(filepath=batch_x, resize=self.dim)
        batch_y = read(filepath=batch_y, resize=self.dim)

        shape_x = (self.batch_size, self.dim, self.dim, self.n_channels)
        shape_y = (self.batch_size, self.dim, self.dim, self.n_channels)
        batch_x = np.reshape(batch_x, shape_x)
        batch_y = np.reshape(batch_y, shape_y)

        return batch_x, batch_y
